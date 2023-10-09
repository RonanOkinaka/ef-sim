//! File where the final particle simulation code lives.

use crate::circle::*;
use crate::line::*;
use crate::render_util::*;
use crate::update_queue::*;
use std::sync::{Arc, RwLock};

use util::math::Point;

/// The method by which one may alter the ParticleSimulation state.
#[derive(Clone)]
pub struct ParticleSimSender {
    particle_sender: ParticleSender,
    charge_updates: UpdateQueue<(Point, f32, f32)>,
}

/// Helper struct to multiplex charge buffer updates between the particle
/// and circle renderers (they are similar in structure).
struct ParticleUpdater<'a, 'b, 'c> {
    line_updates: VecToWgpuBufHelper<'a, Charge>,
    circle_updates: CircleUpdater<'a, 'b, 'c>,
}

pub fn particle_simulation(render_graph: &mut RenderGraph) -> ParticleSimSender {
    // Create the particle renderer
    let particle_render = ParticleRenderer::new(render_graph);
    let particle_sender = ParticleSender::from_renderer(&particle_render);

    // Create the circle renderer
    let circle_render = Arc::new(RwLock::new(CircleRenderer::new(
        render_graph,
        particle_render.limits().max_num_charges,
    )));
    let circle_updater = circle_render.clone();

    // Prepare the shared update queue
    let charge_updates = UpdateQueue::<(Point, f32, f32)>::with_limit(
        particle_render.limits().max_num_charges as usize,
    );

    // Create the particle render passes
    let particle_render = Arc::new(RwLock::new(particle_render));
    let particle_updater = particle_render.clone();

    let popper = particle_render.clone();
    let pop_handle = render_graph.add_pass(Box::new(move |context| {
        popper.read().unwrap().do_pop_pass(context);
    }));

    let particle_render_handle = render_graph.add_pass(Box::new(move |context| {
        particle_render.read().unwrap().do_render_pass(context);
    }));

    // Create the circle render pass
    let circle_render_handle = render_graph.add_pass(Box::new(move |context| {
        circle_render.read().unwrap().render(context);
    }));

    // Create the shared update pass
    let charge_updates_cloned = charge_updates.clone();
    let update_handle = render_graph.add_pass(Box::new(move |context| {
        let mut particle_updater = particle_updater.write().unwrap();
        let mut circle_updater = circle_updater.write().unwrap();

        {
            charge_updates_cloned.apply_updates(&mut ParticleUpdater {
                line_updates: particle_updater.get_charge_updater(context),
                circle_updates: circle_updater.get_updater(context),
            });
        }
        particle_updater.do_update_pass(context);
    }));

    // These won't run concurrently due to the RwLock, so we might as well
    // prevent the threads from blocking
    render_graph.set_sequence(update_handle, RenderOrder::RunsBefore, pop_handle);
    render_graph.set_sequence(
        update_handle,
        RenderOrder::RunsBefore,
        particle_render_handle,
    );
    render_graph.set_sequence(update_handle, RenderOrder::RunsBefore, circle_render_handle);

    // These two can run concurrently, however
    render_graph.set_sequence(
        pop_handle,
        RenderOrder::SubmitsBefore,
        particle_render_handle,
    );

    // Configure the global draw sequence
    render_graph.set_sequence(
        particle_render_handle,
        RenderOrder::SubmitsBefore,
        circle_render_handle,
    );

    ParticleSimSender {
        particle_sender,
        charge_updates,
    }
}

impl ParticleSimSender {
    pub fn push_charge(
        &self,
        pos: Point,
        charge: f32,
        radius: f32,
    ) -> Result<u32, (Point, f32, f32)> {
        match self.charge_updates.push((pos, charge, radius)) {
            Ok(index) => Ok(index as u32),
            Err(value) => Err(value),
        }
    }

    pub fn pop_charge(&self, index: u32) -> Result<(), u32> {
        match self.charge_updates.pop(index as usize) {
            Ok(..) => Ok(()),
            Err(index) => Err(index as u32),
        }
    }

    pub fn set_num_curves(&self, num_curves: u32) {
        self.particle_sender.set_num_curves(num_curves);
    }

    pub fn set_particle_lifetime(&self, lifetime_s: f32) {
        self.particle_sender.set_particle_lifetime(lifetime_s)
    }
}

impl<'a, 'b, 'c> PushAndSwapPop<(Point, f32, f32)> for ParticleUpdater<'a, 'b, 'c> {
    fn push(&mut self, value: (Point, f32, f32)) {
        self.circle_updates.push((value.0, value.2));
        self.line_updates.push(Charge {
            pos: value.0,
            charge: value.1,
            _pad: 0,
        });
    }

    fn pop(&mut self, index: usize) {
        self.circle_updates.pop(index);
        self.line_updates.pop(index);
    }

    fn len(&self) -> usize {
        self.line_updates.len()
    }

    fn finalize(&self, result: UpdateResult) {
        self.circle_updates.finalize(result);
        self.line_updates.finalize(result);
    }
}
