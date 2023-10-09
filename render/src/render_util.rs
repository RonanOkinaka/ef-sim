//! Various utilities for rendering.

use rayon::{Scope, ThreadPool, ThreadPoolBuilder};
use std::collections::HashSet;
use std::mem::swap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use crate::window::Window;

pub type Pass = Box<dyn Fn(&RenderContext) + Send>;

/// Render graph based on traversing a DAG in topo-sort order.
/// Concurrency achieved with rayon.
pub struct RenderGraph {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface,

    render_nodes: Vec<RenderNode>,
    bootstrap_nodes: HashSet<usize>, // TODO: This is too heavy for its use
    finalize_handle: usize,

    thread_pool: ThreadPool,
    tick_over: Arc<AtomicBool>,

    command_buf_tx: mpsc::Sender<CommandBufData>,
    command_buf_rx: mpsc::Receiver<CommandBufData>,
}

/// The WGPU context + a view to the framebuffer + the thread scope
/// from rayon to allow internal threading.
/// Please use context.submit() instead of queue.submit() when an
/// order is RenderOrder::SubmitsBefore.
pub struct RenderContext<'tick, 'inner> {
    pub adapter: &'tick wgpu::Adapter,
    pub device: &'tick wgpu::Device,
    pub queue: &'tick wgpu::Queue,
    pub surface: &'tick wgpu::Surface,
    pub frame_view: &'tick wgpu::TextureView,
    pub thread_scope: &'inner Scope<'tick>,

    command_buf_tx: &'inner mpsc::Sender<CommandBufData>,
}

/// The semantic ordering of two passes.
pub enum RenderOrder {
    /// Denotes a strict dependency ("strong edge") between two passes.
    /// The second will not start before the first finishes.
    RunsBefore,
    /// Weak edge: two passes can run asynchronously as long as their
    /// command buffers are submitted in dependency-order.
    /// This is opaque.
    SubmitsBefore,
}

struct RenderContextInner<'tick> {
    // TODO: Unfortunate duplication...
    adapter: &'tick wgpu::Adapter,
    device: &'tick wgpu::Device,
    queue: &'tick wgpu::Queue,
    surface: &'tick wgpu::Surface,
    frame_view: &'tick wgpu::TextureView,
}

struct RenderNode {
    callback: Mutex<Pass>,
    outgoing: Vec<usize>,

    submitter: Option<usize>,
    command_buf_tx: mpsc::Sender<CommandBufData>,

    indegree_start: usize,
    indegree_running: AtomicUsize,
}

enum CommandBufData {
    FullBuffer(wgpu::CommandBuffer),
    EmptyBuffer,
}

impl RenderGraph {
    /// Add a callback to be run once per tick().
    /// The callback is provided an updated RenderContext every frame.
    /// Returns a key to later reference the pass.
    pub fn add_pass(&mut self, callback: Pass) -> usize {
        // Add the pass
        let handle = self.add_pass_inner(callback, Vec::new(), 0);

        // Ensure that our final task runs after it
        self.set_run_sequence(handle, self.finalize_handle);

        handle
    }

    /// Provided two passes and an ordering, enforce that order in future tick() calls.
    /// Uses the keys provided by add_pass().
    /// Does not check for duplicate or cyclical dependencies.
    pub fn set_sequence(&mut self, before: usize, order: RenderOrder, after: usize) {
        match order {
            RenderOrder::RunsBefore => {
                self.set_run_sequence(before, after);
            }
            RenderOrder::SubmitsBefore => {
                self.set_submit_sequence(before, after);
            }
        }
    }

    /// Traverses the DAG in dependency-order via concurrent topo-sort, providing
    /// RenderContext to callbacks along the way and enforcing order from set_sequence().
    pub fn tick(&mut self) {
        // Prepare the current frame
        // Note that if this fails due to timeout, it's almost certainly deadlock
        let curr_frame = self
            .surface
            .get_current_texture()
            .expect("Failed to retrieve swapchain texture");
        let frame_view = curr_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let context = &RenderContextInner {
            adapter: &self.adapter,
            device: &self.device,
            queue: &self.queue,
            surface: &self.surface,
            frame_view: &frame_view,
        };

        let render_nodes = self.render_nodes.as_ref();

        self.tick_over.store(false, Ordering::Relaxed);

        // Do the actual render work
        self.thread_pool.in_place_scope(|scope| {
            // Start at the nodes known to have indegree 0
            for node_index in self.bootstrap_nodes.iter() {
                scope.spawn(move |thread_scope| {
                    Self::topo_visit(*node_index, render_nodes, context, thread_scope);
                });
            }

            // Try to submit command buffers as they come in
            while !self.tick_over.load(Ordering::Acquire) {
                // There will always be at least one
                let cmd_bufs = [self.command_buf_rx.recv().unwrap()]
                    .into_iter()
                    // Grab the others that are also available
                    .chain(self.command_buf_rx.try_iter())
                    .filter_map(|buf_data| match buf_data {
                        CommandBufData::FullBuffer(cmd_buf) => Some(cmd_buf),
                        CommandBufData::EmptyBuffer => None,
                    });

                self.queue.submit(cmd_bufs);
            }
        });

        // Collect any stray command buffers
        let commands = self
            .command_buf_rx
            .try_iter()
            .filter_map(|buf_data| match buf_data {
                CommandBufData::FullBuffer(cmd_buf) => Some(cmd_buf),
                CommandBufData::EmptyBuffer => None,
            });
        self.queue.submit(commands);

        curr_frame.present();
    }

    /// Create a RenderGraph from a Window.
    pub async fn from_window(window: &Window) -> RenderGraph {
        let instance = wgpu::Instance::default();

        // Unfortunate but unavoidable
        let surface = unsafe { instance.create_surface(window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let (width, height) = window.get_size();
        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: swapchain_capabilities.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        // Prepare threading
        let num_threads = std::thread::available_parallelism()
            .unwrap_or(std::num::NonZeroUsize::new(4).unwrap())
            .get();
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let (command_buf_tx, command_buf_rx) = mpsc::channel();

        let mut ret = Self {
            adapter,
            device,
            queue,
            surface,
            render_nodes: Vec::new(),
            bootstrap_nodes: HashSet::new(),
            finalize_handle: 0,
            thread_pool,
            tick_over: Arc::new(AtomicBool::new(true)),
            command_buf_tx,
            command_buf_rx,
        };

        // This callback runs after everything else to mark the end of the tick
        let tick_over_produce = ret.tick_over.clone();
        ret.finalize_handle = ret.add_pass_inner(
            Box::new(move |context| {
                tick_over_produce.store(true, Ordering::Release);
                context.submit_empty().unwrap();
            }),
            Vec::new(),
            0,
        );

        ret
    }

    fn add_pass_inner(&mut self, callback: Pass, outgoing: Vec<usize>, indegree: usize) -> usize {
        let index = self.render_nodes.len();

        // Add node to the collection
        self.render_nodes.push(RenderNode {
            callback: Mutex::new(callback),
            outgoing,
            submitter: None,
            command_buf_tx: self.command_buf_tx.clone(),
            indegree_start: indegree,
            indegree_running: AtomicUsize::new(indegree),
        });

        // Mark that the node currently has indegree 0
        self.bootstrap_nodes.insert(index);

        index
    }

    fn set_submit_sequence(&mut self, before: usize, after: usize) {
        // How does this work?
        // It doesn't matter whether <before> or <after> runs first; the
        // submitter will run later than both (and the <before> submitter, if
        // one exists), pushing its command buffer(s) at that time.
        let submitter_index = self.get_submitter(after);
        self.set_run_sequence(before, submitter_index);
    }

    fn get_submitter(&mut self, index: usize) -> usize {
        let submitter_handle = self.render_nodes.len(); // TODO: Don't hard-code this
        let mut outgoing;
        let submit_rx;
        {
            let node = &mut self.render_nodes[index];

            // If we already have one, do nothing
            if let Some(submitter_handle) = node.submitter {
                return submitter_handle;
            }

            // Insert an intermediary channel after the original
            (node.command_buf_tx, submit_rx) = mpsc::channel();

            // Old: (node) -> (dependencies)
            // New: (node) -> (submitter) -> (dependencies)
            outgoing = vec![submitter_handle];
            swap(&mut outgoing, &mut node.outgoing);

            node.submitter = Some(submitter_handle);
        }

        self.add_pass_inner(
            Box::new(move |context| {
                // All the submitter does is submit data on behalf of <node>
                while let Ok(cmd_buf) = submit_rx.try_recv() {
                    context.command_buf_tx.send(cmd_buf).unwrap();
                }
            }),
            outgoing,
            1,
        )
    }

    /// Set the node <after> to run strictly later than the latest associated
    /// node of <before> (the node or its submitter).
    fn set_run_sequence(&mut self, before: usize, after: usize) {
        let finalizer_handle = self.finalize_handle;
        let first_node = self.get_constraint_node(before);

        // Add outgoing edge from <before>
        if first_node.outgoing.len() == 1 && first_node.outgoing[0] == finalizer_handle {
            // "After-ness" is transitive, so replace the edge to the finalizer
            first_node.outgoing[0] = after;

            let finalizer = &mut self.render_nodes[finalizer_handle];
            finalizer.indegree_running.fetch_sub(1, Ordering::Release);
            finalizer.indegree_start -= 1;
        } else {
            // Otherwise, just push the edge
            first_node.outgoing.push(after);
        }

        // Add incoming edge to <after>
        self.bootstrap_nodes.remove(&after);
        self.render_nodes[after].indegree_start += 1;
        self.render_nodes[after]
            .indegree_running
            .fetch_add(1, Ordering::Release);
    }

    /// Retrieve the node with stronger ordering constraints: that is, the
    /// node or its associated submitter.
    fn get_constraint_node(&mut self, index: usize) -> &mut RenderNode {
        match self.render_nodes[index].submitter {
            Some(index) => &mut self.render_nodes[index],
            None => &mut self.render_nodes[index],
        }
    }

    fn topo_visit<'tick>(
        node_index: usize,
        render_nodes: &'tick Vec<RenderNode>,
        context: &'tick RenderContextInner,
        thread_scope: &Scope<'tick>,
    ) {
        let node = &render_nodes[node_index];

        // Load the atomic marker to ensure that side-effects are seen
        // before the callback begins
        node.indegree_running.load(Ordering::Acquire);

        // Run the render step
        // TODO: The mutex does more for us than we need; one atomic bool
        // should be enough
        match node.callback.try_lock() {
            Ok(callback) => (*callback)(&RenderContext {
                adapter: context.adapter,
                device: context.device,
                queue: context.queue,
                surface: context.surface,
                frame_view: context.frame_view,
                thread_scope,
                command_buf_tx: &node.command_buf_tx,
            }),
            Err(..) => panic!("Callbacks should not be invoked twice in one tick"),
        }

        // Check to see which of the next nodes are unblocked
        for next_index in node.outgoing.iter() {
            let next_node = &render_nodes[*next_index];

            // If we were the last dependency, start the next task
            if next_node.indegree_running.fetch_sub(1, Ordering::Release) == 1 {
                // Reset the counter for the next node
                next_node
                    .indegree_running
                    .store(next_node.indegree_start, Ordering::Release);

                // Start the next task
                thread_scope.spawn(move |thread_scope| {
                    Self::topo_visit(*next_index, render_nodes, context, thread_scope);
                });
            }
        }
    }
}

impl<'tick, 'inner> RenderContext<'tick, 'inner> {
    pub fn submit(&self, cmd_buf: wgpu::CommandBuffer) -> Result<(), wgpu::CommandBuffer> {
        match self
            .command_buf_tx
            .send(CommandBufData::FullBuffer(cmd_buf))
        {
            Ok(..) => Ok(()),
            Err(mpsc::SendError(CommandBufData::FullBuffer(cmd_buf))) => Err(cmd_buf),
            _ => panic!("submit() should not send empty buffers"),
        }
    }

    pub fn submit_empty(&self) -> Result<(), ()> {
        match self.command_buf_tx.send(CommandBufData::EmptyBuffer) {
            Ok(..) => Ok(()),
            Err(..) => Err(()),
        }
    }
}
