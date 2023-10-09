//! Utility for dealing with many writers pushing to and popping from
//! one buffer when order does not matter.

use bytemuck::{cast_slice, Pod};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Identifies an object that supports pushing, erasing by swap, and
/// retrieving its size (obvious example: Vec<>)
pub trait PushAndSwapPop<T> {
    /// Push value into container.
    fn push(&mut self, value: T);

    /// Pop value from container by swapping to back.
    fn pop(&mut self, index: usize);

    /// Return the length of the container.
    fn len(&self) -> usize;

    /// Any operations that must be done after CPU-side updates.
    fn finalize(&self, result: UpdateResult);
}

/// Abstraction over a shared buffer where updates are queued to be done
/// at a later time, with compaction.
#[derive(Clone)]
pub struct UpdateQueue<T> {
    // TODO: This could be lock-free (albeit much more complex)
    data: Arc<Mutex<UpdateQueueInner<T>>>,
    limit: usize,
}

pub enum UpdateType<T> {
    Push(T),
    SwapPop(usize),
}

#[derive(Debug, Clone, Copy)]
pub enum UpdateResult {
    Same,
    SizeOnly(usize),
    Range(usize, usize),
}

/// An admittedly not very general solution to tightly update WGPU buffers
/// given a mirrored version on the CPU.
pub struct VecToWgpuBufHelper<'a, T> {
    /// Offset of the data in bytes.
    pub data_off: Option<u64>,
    /// Offset of the size information in bytes.
    pub size_off: Option<u64>,
    /// Data vector.
    pub vec: &'a mut Vec<T>,
    /// Data buffer.
    pub buf: &'a wgpu::Buffer,
    /// WGPU's queue.
    pub queue: &'a wgpu::Queue,
}

struct UpdateQueueInner<T> {
    updates: VecDeque<UpdateType<T>>,
    target_size: usize,

    index_to_key: Vec<usize>,
    key_to_index: HashMap<usize, usize>,
    key_counter: usize,
}

impl<T> UpdateQueue<T> {
    /// Represent a buffer with some maximum size.
    pub fn with_limit(limit: usize) -> Self {
        let inner = UpdateQueueInner {
            updates: VecDeque::new(),
            target_size: 0,
            index_to_key: Vec::with_capacity(limit),
            key_to_index: HashMap::new(),
            key_counter: 0,
        };

        Self {
            data: Arc::new(Mutex::new(inner)),
            limit,
        }
    }

    /// Push a new value to the buffer (virtually).
    pub fn push(&self, value: T) -> Result<usize, T> {
        let mut inner = self.data.lock().unwrap();

        // Push to the end
        let new_size = inner.target_size + 1;
        if new_size <= self.limit {
            // Allocate a new key
            let key = inner.key_counter;
            inner.key_counter += 1;

            // Push it into our inverse map
            let back = inner.index_to_key.len();
            inner.index_to_key.push(key);

            // Write it into the update list and index map
            inner.key_to_index.insert(key, back);
            inner.updates.push_back(UpdateType::Push(value));
            inner.target_size = new_size;

            Ok(key)
        } else {
            Err(value)
        }
    }

    /// Pop a value from the buffer.
    pub fn pop(&self, key: usize) -> Result<(), usize> {
        let mut inner = self.data.lock().unwrap();

        // This key becomes invalid
        match inner.key_to_index.remove(&key) {
            Some(index) => {
                inner.target_size -= 1;

                let swap_key = *inner.index_to_key.last().unwrap();
                inner.index_to_key.swap_remove(index);

                inner.updates.push_back(UpdateType::SwapPop(index));

                if let Some(mapped_index) = inner.key_to_index.get_mut(&swap_key) {
                    *mapped_index = index;
                }

                Ok(())
            }
            None => Err(key), // Double-delete!
        }
    }

    /// Iterate over the updates, exactly once. Cleared after.
    pub fn for_each<F>(&self, callback: F)
    where
        F: FnMut(UpdateType<T>),
    {
        // TODO: Iterator with parking_lot?
        self.data
            .lock()
            .unwrap()
            .updates
            .drain(..)
            .for_each(callback);
    }

    /// Apply the queued updates to a Vec<>, and return the bounds that were
    /// updated [min, max). Cleared after.
    /// Calls finalize() on the provided container, but still returns
    /// UpdateResult if it is convenient.
    pub fn apply_updates<U>(&self, container: &mut U) -> UpdateResult
    where
        U: PushAndSwapPop<T>,
    {
        let mut min = usize::MAX;
        let mut max = usize::MIN;

        self.for_each(|update| match update {
            UpdateType::Push(value) => {
                let index = container.len();
                min = min.min(index);
                max = max.max(index);

                container.push(value);
            }
            UpdateType::SwapPop(index) => {
                min = min.min(index);
                max = max.max(index);

                container.pop(index);
            }
        });

        let len = container.len();
        let result;

        if min == usize::MAX {
            result = UpdateResult::Same;
        } else if min >= len {
            result = UpdateResult::SizeOnly(len);
        } else {
            result = UpdateResult::Range(min, (max + 1).min(len));
        }

        container.finalize(result);
        result
    }
}

impl<'a, T> PushAndSwapPop<T> for VecToWgpuBufHelper<'a, T>
where
    T: Pod,
{
    fn push(&mut self, value: T) {
        self.vec.push(value);
    }

    fn pop(&mut self, index: usize) {
        self.vec.swap_remove(index);
    }

    fn len(&self) -> usize {
        self.vec.len()
    }

    fn finalize(&self, result: UpdateResult) {
        match result {
            UpdateResult::Range(min, max) => {
                // If we have data to write, do so
                if let Some(data_off) = self.data_off {
                    self.queue.write_buffer(
                        self.buf,
                        data_off + (min * std::mem::size_of::<T>()) as u64,
                        cast_slice(&self.vec[min..max]),
                    );
                }
            }
            UpdateResult::SizeOnly(..) => { /* Just update the size */ },
            UpdateResult::Same => return, // Early return if nothing happened
        }

        // Update the buffer size
        if let Some(size_off) = self.size_off {
            self.queue
                .write_buffer(self.buf, size_off, cast_slice(&[self.vec.len() as u32]));
        }
    }
}

impl<T> PushAndSwapPop<T> for Vec<T> {
    fn push(&mut self, value: T) {
        self.push(value);
    }

    fn pop(&mut self, index: usize) {
        self.swap_remove(index);
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn finalize(&self, _: UpdateResult) {
        // No-op
    }
}

#[cfg(test)]
mod test_update_queue {
    use super::*;

    #[test]
    fn test_do_nothing() -> Result<(), &'static str> {
        let update_queue = UpdateQueue::with_limit(1);
        let mut vec = vec![0, 1, 2];

        match update_queue.apply_updates(&mut vec) {
            UpdateResult::Same => {}
            _ => return Err("Empty update queues should have no effect"),
        }

        Ok(())
    }

    #[test]
    fn test_push() -> Result<(), &'static str> {
        let update_queue = UpdateQueue::with_limit(1);
        let mut vec = vec![0, 1, 2];

        if update_queue.push(3).is_err() {
            return Err("Failed to push");
        }

        match update_queue.apply_updates(&mut vec) {
            UpdateResult::Range(3, 4) => {}
            _ => return Err("Push bounds are incorrect"),
        }

        if vec != vec![0, 1, 2, 3] {
            return Err("Updates applied incorrectly");
        }

        Ok(())
    }

    #[test]
    fn test_multi_push() -> Result<(), &'static str> {
        let update_queue = UpdateQueue::with_limit(5);
        let mut vec = vec![0, 1, 2];

        update_queue.push(3).unwrap();
        update_queue.push(4).unwrap();
        update_queue.push(5).unwrap();
        update_queue.push(6).unwrap();
        update_queue.push(7).unwrap();

        match update_queue.apply_updates(&mut vec) {
            UpdateResult::Range(3, 8) => {}
            _ => return Err("Push bounds are incorrect"),
        }

        if vec != vec![0, 1, 2, 3, 4, 5, 6, 7] {
            return Err("Pushes should be applied in order");
        }

        Ok(())
    }

    #[test]
    fn test_push_limit() -> Result<(), &'static str> {
        let update_queue = UpdateQueue::with_limit(3);
        let mut vec = vec![0, 1, 2];

        update_queue.push(3).unwrap();
        update_queue.push(4).unwrap();
        update_queue.push(5).unwrap();

        if update_queue.push(6).is_ok() || update_queue.push(7).is_ok() {
            return Err("Should reject pushes past buffer limit");
        }

        match update_queue.apply_updates(&mut vec) {
            UpdateResult::Range(3, 6) => {}
            _ => return Err("Push bounds are incorrect"),
        }

        if vec != vec![0, 1, 2, 3, 4, 5] {
            return Err("Updates applied incorrectly");
        }

        Ok(())
    }

    #[test]
    fn test_pop() -> Result<(), &'static str> {
        let update_queue = UpdateQueue::with_limit(5);
        let mut vec: Vec<u32> = Vec::new();

        if update_queue.pop(0).is_ok() {
            return Err("Should reject invalid key");
        }

        let pop0 = update_queue.push(0).unwrap();

        if update_queue.pop(pop0).is_err() {
            return Err("Failed to pop");
        }

        match update_queue.apply_updates(&mut vec) {
            UpdateResult::SizeOnly(0) => {}
            _ => return Err("Failed to classify empty buffer"),
        }

        if !vec.is_empty() {
            return Err("Updates applied incorrectly");
        }

        Ok(())
    }

    #[test]
    fn test_pop_back() -> Result<(), &'static str> {
        let update_queue = UpdateQueue::with_limit(5);
        let mut vec: Vec<u32> = Vec::new();

        update_queue.push(0).unwrap();
        let pop1 = update_queue.push(1).unwrap();
        update_queue.apply_updates(&mut vec);

        update_queue.pop(pop1).unwrap();
        let result = update_queue.apply_updates(&mut vec);

        match result {
            UpdateResult::SizeOnly(1) => {}
            _ => return Err("Failed to adjust size only when popping from back"),
        }

        if vec != vec![0] {
            return Err("Failed to pop from back of vec");
        }

        Ok(())
    }

    #[test]
    fn test_double_delete() -> Result<(), &'static str> {
        let update_queue = UpdateQueue::with_limit(5);

        let pop0 = update_queue.push(0).unwrap();
        update_queue.pop(pop0).unwrap();

        if update_queue.pop(pop0).is_ok() {
            return Err("Failed to catch double-delete");
        }

        Ok(())
    }

    #[test]
    fn test_multi_pop() -> Result<(), &'static str> {
        let update_queue = UpdateQueue::with_limit(5);
        let mut vec: Vec<u32> = Vec::new();

        let _0 = update_queue.push(0).unwrap();
        let pop1 = update_queue.push(1).unwrap();
        let _2 = update_queue.push(2).unwrap();
        let pop3 = update_queue.push(3).unwrap();

        if update_queue.pop(pop1).is_err() || update_queue.pop(pop3).is_err() {
            return Err("Failed to pop valid key");
        }

        match update_queue.apply_updates(&mut vec) {
            UpdateResult::Range(0, 2) => {}
            blah => {
                println!("{:#?}", blah);
                return Err("Improper update bounds");
            }
        }

        if vec != vec![0, 2] {
            return Err("Updates applied incorrectly");
        }

        Ok(())
    }
}
