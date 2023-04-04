//! In this file lies a simple implementation of an AABB tree.
//! In particular, it includes a bare-bones AABB structure, and a tree which
//! supports insertion, deletion, (simple) retrieval and basic collision queries.

/// Simple implementation of an axis-aligned bounding box (AABB).
#[derive(Clone, Copy, Debug)]
pub struct Rect<F> {
    /// Leftmost edge
    pub x0: F,
    /// Rightmost edge
    pub x1: F,
    /// Bottommost edge
    pub y0: F,
    /// Topmost edge
    pub y1: F,
}

impl<F> Rect<F>
where
    F: Copy,
{
    /// Creates a new AABB with axis-aligned edges at `x0`, `x1` and `y0`, `y1`.
    /// It is assumed that `x0 < x1` and `y0 < y1`.
    #[inline]
    pub const fn new(x0: F, x1: F, y0: F, y1: F) -> Self {
        Self { x0, x1, y0, y1 }
    }

    /// Creates an AABB that strictly bounds the two parameters
    #[inline]
    pub fn as_merge(mut r1: Self, r2: Self) -> Self
    where
        F: std::cmp::PartialOrd,
    {
        r1.merge_with(r2);
        r1
    }

    /// Ensures that the AABB represented by `self` can bound the parameter.
    ///
    /// # Panics
    /// Panics if any edges are incomparable (e.g. NaN for floats).
    #[inline]
    pub fn merge_with(&mut self, r2: Self)
    where
        F: std::cmp::PartialOrd,
    {
        // TODO: Don't just give up when we have a NaN
        let assume_cmp = |a: &F, b: &F| a.partial_cmp(b).unwrap();

        self.x0 = std::cmp::min_by(self.x0, r2.x0, assume_cmp);
        self.x1 = std::cmp::max_by(self.x1, r2.x1, assume_cmp);
        self.y0 = std::cmp::min_by(self.y0, r2.y0, assume_cmp);
        self.y1 = std::cmp::max_by(self.y1, r2.y1, assume_cmp);
    }

    /// Determines whether two AABBs overlap.
    #[inline]
    pub fn overlaps(&self, r2: Self) -> bool
    where
        F: std::cmp::PartialOrd,
    {
        (self.x0 <= r2.x1) && (r2.x0 <= self.x1) && (self.y0 <= r2.y1) && (r2.y0 <= self.y1)
    }

    /// Calculates the area of the box.
    fn area(&self) -> F
    where
        F: std::ops::Sub<Output = F> + std::ops::Mul<Output = F>,
    {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }
}

/// Stores the node's payload: either indices referring to two children or user data (`T`).
/// Because they share space, smaller structs work better (indirection could be necessary).
/// It is also worth noting that the tree takes ownership, so `Rc<T>` may be required.
#[derive(Debug)]
enum NodeData<T> {
    /// Short for "junction," stores indices for two other nodes.
    Junc(usize, usize),
    /// Leaf node, stores user data.
    Leaf(T),
}

/// One single node in the AABB tree, with all necessary information.
#[derive(Debug)]
struct TreeNode<T, F> {
    /// The payload, either two children or some mapped type.
    data: NodeData<T>,
    /// The fitted AABB
    bounds: Rect<F>,
    /// This node's parent (will be INVALID_INDEX if root).
    parent: usize,
}

/// Represents the AABB tree itself, where `T` is the mapped type and
/// `F` is the coordinate type (e.g. `f32`).
#[derive(Debug)]
pub struct AabbTree<T, F> {
    /// We store all nodes in a vector, referred to by their indices.
    nodes: Vec<TreeNode<T, F>>,
    /// A free-list of unused positions in the vector (because I'm lazy).
    free: Vec<usize>,
    /// Index of the root node in the vector.
    root: usize,
}

/// Refers to one leaf node in the associated AabbTree, with few safety features.
#[derive(Copy, Clone)]
pub struct NodeRef {
    /// The index of the node.
    index: usize,
}

/// An iterator-like type that performs DFS on the tree according to some query.
pub struct AabbOverlapIter<'a, T, F, Q> {
    /// A callable query taking one Rect, determines whether to continue down the tree.
    query: Q,
    /// A stack merely to make the DFS easier to write.
    /// Note that it isn't strictly necessary and will be refactored at some point in the future.
    stack: Vec<usize>,
    /// A reference to the node vector.
    container: &'a Vec<TreeNode<T, F>>,
}

const INVALID_INDEX: usize = usize::MAX;

impl<T, F> AabbTree<T, F>
where
    F: Copy + std::cmp::PartialOrd + std::ops::Sub<Output = F> + std::ops::Mul<Output = F>,
{
    /// Creates a new tree with enough space for the number of nodes provided.
    pub fn with_capacity(num_leaves: usize) -> Self {
        // Structured as full binary tree, so we have exactly one fewer
        // junctions than we do leaves (hence, we multiply by 2)
        Self {
            nodes: Vec::with_capacity(num_leaves * 2),
            free: Vec::new(),
            root: INVALID_INDEX,
        }
    }

    /// Inserts new user data with an associated bounding box, then returns
    /// a "reference" which can be used to retrieve or delete the data.
    pub fn insert(&mut self, bounds: Rect<F>, data: T) -> NodeRef {
        // If this is our first node, just insert it and leave
        if Self::is_invalid_index(self.root) {
            self.root = self.create_leaf(data, bounds, INVALID_INDEX);
            return NodeRef { index: self.root };
        }

        // Otherwise, insert both a junction and a leaf
        let (junc, leaf) = self.create_leaf_junc(data, bounds);

        // Now, try to find the best sibling node for our new leaf
        let mut new_bounds = Rect::as_merge(self.nodes[self.root].bounds, bounds);
        let mut sibling = self.root;
        while let NodeData::Junc(left_child, right_child) = self.nodes[sibling].data {
            // Update bounding boxes on the way down
            self.nodes[sibling].bounds = new_bounds;

            let (better_child, better_bounds) =
                self.choose_better_child(bounds, left_child, right_child);

            new_bounds = better_bounds;
            sibling = better_child;
        }

        // Once we've found the ideal sibling, insert junction "there" and
        // push the leaves down below
        self.steal_parents(junc, sibling);
        let junc_bounds = Rect::as_merge(self.nodes[sibling].bounds, bounds);
        self.nodes[junc].bounds = junc_bounds;
        self.nodes[junc].data = NodeData::Junc(sibling, leaf);
        self.nodes[sibling].parent = junc;

        NodeRef { index: leaf }
    }

    /// Retrieves the data from a particular insertion.
    pub fn get(&self, index: NodeRef) -> &T {
        match self.nodes[index.index].data {
            NodeData::Leaf(ref val) => val,
            NodeData::Junc(..) => panic!("Should not get() junction nodes"),
        }
    }

    /// Mutably retrieves the data from a particular insertion.
    pub fn get_mut(&mut self, index: NodeRef) -> &mut T {
        match self.nodes[index.index].data {
            NodeData::Leaf(ref mut val) => val,
            NodeData::Junc(..) => panic!("Should not get() junction nodes"),
        }
    }

    /// Iterates over all leaf nodes matching a particular query.
    ///
    /// In particular, the query is a callable that takes one `Rect<F>` as input and
    /// returns a boolean value.
    /// A leaf is only returned if the query returns true for all junctions leading up to
    /// the leaf AND the leaf itself.
    /// The characteristic property of the AABB tree is that all child nodes have an AABB
    /// strictly contained within their parent.
    pub fn iter_query<Q>(&self, query: Q) -> AabbOverlapIter<'_, T, F, Q>
    where
        Q: FnMut(Rect<F>) -> bool,
    {
        let stack = match self.root {
            INVALID_INDEX => Vec::new(),
            root => vec![root],
        };

        AabbOverlapIter {
            query,
            stack,
            container: &self.nodes,
        }
    }

    /// The most useful application of the query iteration, returns all leaves that overlap
    /// the provided bounds.
    pub fn iter_overlap(
        &self,
        bounds: Rect<F>,
    ) -> AabbOverlapIter<'_, T, F, impl FnMut(Rect<F>) -> bool> {
        self.iter_query(move |rect: Rect<F>| rect.overlaps(bounds))
    }

    pub fn erase(&mut self, index: NodeRef) {
        // Fetch the parent and sibling of the node (if they exist)
        let parent = self.nodes[index.index].parent;

        if Self::is_invalid_index(parent) {
            self.root = INVALID_INDEX;
            return;
        }

        let sibling = match self.nodes[parent].data {
            NodeData::Junc(left, right) => {
                if left == index.index {
                    right
                } else {
                    left
                }
            }
            NodeData::Leaf(..) => panic!("Leaf nodes should not have children"),
        };

        // Promote sibling to parent's position
        self.steal_parents(sibling, parent);

        // Update bounding boxes up the tree
        let mut node = self.nodes[sibling].parent;
        while !Self::is_invalid_index(node) {
            match self.nodes[node].data {
                NodeData::Junc(left, right) => {
                    self.nodes[node].bounds =
                        Rect::as_merge(self.nodes[left].bounds, self.nodes[right].bounds);
                }
                NodeData::Leaf(..) => panic!("Leaf nodes should not have children"),
            }

            node = self.nodes[node].parent;
        }

        // Simply mark nodes as unused without "deleting" them
        self.free.push(index.index);
        self.free.push(parent);
        self.nodes[index.index].data = NodeData::Junc(INVALID_INDEX, INVALID_INDEX);
        self.nodes[parent].data = NodeData::Junc(INVALID_INDEX, INVALID_INDEX);
    }

    /// Inserts one new node into the tree, free-list first.
    fn push_node(&mut self, data: NodeData<T>, bounds: Rect<F>, parent: usize) -> usize {
        match self.free.pop() {
            None => {
                self.nodes.push(TreeNode {
                    data,
                    bounds,
                    parent,
                });
                self.nodes.len() - 1
            }
            Some(index) => {
                self.nodes[index] = TreeNode {
                    data,
                    bounds,
                    parent,
                };
                index
            }
        }
    }

    /// Pushes a leaf into the tree.
    fn create_leaf(&mut self, data: T, bounds: Rect<F>, parent: usize) -> usize {
        self.push_node(NodeData::Leaf(data), bounds, parent)
    }

    /// Pushes both a leaf and an associated junction into the tree.
    fn create_leaf_junc(&mut self, data: T, bounds: Rect<F>) -> (usize, usize) {
        let junc = self.push_node(
            NodeData::Junc(INVALID_INDEX, INVALID_INDEX),
            bounds,
            INVALID_INDEX,
        );
        let leaf = self.create_leaf(data, bounds, junc);

        (junc, leaf)
    }

    /// Splices subtrees, allowing nodes to "steal" the position of others.
    fn steal_parents(&mut self, thief: usize, victim: usize) {
        // Steal the parent from the victim
        let parent = self.nodes[victim].parent;
        self.nodes[thief].parent = parent;

        // Rewire parent to point to new child
        if Self::is_invalid_index(parent) {
            self.root = thief;
        } else {
            match self.nodes[parent].data {
                NodeData::Junc(ref mut left, ref mut right) => {
                    if *left == victim {
                        *left = thief;
                    } else {
                        *right = thief;
                    }
                }
                NodeData::Leaf(..) => panic!("Leaf nodes should not have children"),
            }
        }
    }

    /// Calculates the better child node in which to place incoming bounds.
    fn choose_better_child(
        &self,
        bounds: Rect<F>,
        left_child: usize,
        right_child: usize,
    ) -> (usize, Rect<F>) {
        // Calculate the candidate AABBs for both children
        let left_bounds = Rect::as_merge(bounds, self.nodes[left_child].bounds);
        let right_bounds = Rect::as_merge(bounds, self.nodes[right_child].bounds);

        // Choose the "better" option [in this implementation, the smaller area]
        if left_bounds.area() < right_bounds.area() {
            (left_child, left_bounds)
        } else {
            (right_child, right_bounds)
        }
    }

    /// Returns the validity of the index.
    fn is_invalid_index(index: usize) -> bool {
        index == INVALID_INDEX
    }
}

impl<'a, T, F, Q> Iterator for AabbOverlapIter<'a, T, F, Q>
where
    F: Copy,
    Q: FnMut(Rect<F>) -> bool,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(index) = self.stack.pop() {
            if !(self.query)(self.container[index].bounds) {
                continue;
            }

            match self.container[index].data {
                NodeData::Leaf(ref data) => return Some(data),
                NodeData::Junc(left, right) => {
                    self.stack.push(right);
                    self.stack.push(left);
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod test_aabb_tree {
    use super::*;

    fn gen_default_tree() -> AabbTree<String, f32> {
        AabbTree::with_capacity(10)
    }

    #[test]
    fn test_root_insertion() -> Result<(), &'static str> {
        let mut tree = gen_default_tree();

        let rect = Rect::new(0.0, 1.0, 0.0, 1.0);
        let index = tree.insert(rect, "root".to_owned());

        if tree.nodes.len() == 0 {
            return Err("Tree should not be empty after insertion");
        } else if !tree.nodes[tree.root].bounds.overlaps(rect) {
            return Err("Root AABB should always overlap children");
        } else if tree.get(index) != "root" {
            return Err("Mapped data type should remain accessible");
        }

        Ok(())
    }

    #[test]
    fn test_child_insertion() -> Result<(), &'static str> {
        let mut tree = gen_default_tree();

        let r1 = Rect::new(-1.0, 0.0, 0.0, 1.0);
        let r2 = Rect::new(1.0, 2.0, 0.0, 1.0);
        let r3 = Rect::new(1.5, 2.5, 0.5, 1.5);
        let r4 = Rect::new(-0.75, -0.5, -0.5, 0.5);

        let i1 = tree.insert(r1, "left".to_owned());
        let i2 = tree.insert(r2, "right".to_owned());
        let i3 = tree.insert(r3, "more right".to_owned());
        let i4 = tree.insert(r4, "more left".to_owned());

        if !tree.nodes[tree.root].bounds.overlaps(r1)
            || !tree.nodes[tree.root].bounds.overlaps(r2)
            || !tree.nodes[tree.root].bounds.overlaps(r3)
            || !tree.nodes[tree.root].bounds.overlaps(r4)
        {
            return Err("Root AABB should always overlap children");
        }

        if tree.get(i1) != "left"
            || tree.get(i2) != "right"
            || tree.get(i3) != "more right"
            || tree.get(i4) != "more left"
        {
            return Err("Mapped data type should remain accessible");
        }

        Ok(())
    }

    #[test]
    fn test_data_mutation() -> Result<(), &'static str> {
        let mut tree = gen_default_tree();
        let index = tree.insert(Rect::new(0.0, 1.0, 0.0, 1.0), "old".to_owned());

        *tree.get_mut(index) = "new".to_owned();

        if tree.get(index) == "old" {
            return Err("Mapped data should be mutable");
        }

        Ok(())
    }

    #[test]
    fn test_overlap_iteration() -> Result<(), &'static str> {
        let mut tree = gen_default_tree();

        for _ in tree.iter_overlap(Rect::new(0.0, 1.0, 0.0, 1.0)) {
            return Err("Empty trees should have no overlaps");
        }

        tree.insert(Rect::new(-1.0, 0.0, 0.0, 1.0), "left".to_owned());
        tree.insert(Rect::new(1.0, 2.0, 0.0, 1.0), "right".to_owned());
        tree.insert(Rect::new(1.5, 2.5, 0.5, 1.5), "more right".to_owned());

        let mut hit: Vec<&String> = tree
            .iter_overlap(Rect::new(-1.5, -0.5, 0.5, 1.5))
            .map(|s| s)
            .collect();
        if hit != vec!["left"] {
            return Err("Partial bounding is overlap");
        }

        hit = tree
            .iter_overlap(Rect::new(1.2, 1.3, 0.1, 0.2))
            .map(|s| s)
            .collect();
        if hit != vec!["right"] {
            return Err("Complete bounding is overlap");
        }

        hit = tree
            .iter_overlap(Rect::new(0.75, 1.75, 0.25, 0.75))
            .map(|s| s)
            .collect();
        if (hit != vec!["right", "more right"]) && (hit != vec!["more right", "right"]) {
            return Err("Must return ALL overlaps");
        }

        hit = tree
            .iter_overlap(Rect::new(0.1, 0.5, 0.25, 0.75))
            .map(|s| s)
            .collect();
        if hit.len() != 0 {
            return Err("Should return nothing when there are no overlaps");
        }

        Ok(())
    }

    // This case is too lengthy but the test setup is cumbersome
    #[test]
    fn test_deletion() -> Result<(), &'static str> {
        let mut tree = gen_default_tree();

        let r1 = Rect::new(-1.0, 0.0, 0.0, 1.0);
        let r2 = Rect::new(1.0, 2.0, 0.0, 1.0);
        let r3 = Rect::new(1.5, 2.5, 0.5, 1.5);
        let r4 = Rect::new(-0.75, -0.5, -0.5, 0.5);

        let i1 = tree.insert(r1, "left".to_owned());

        tree.erase(i1);
        for _ in tree.iter_overlap(Rect::new(-0.5, 0.5, -0.5, 0.5)) {
            return Err("Erased nodes should not be checked for overlap");
        }

        let i1 = tree.insert(r1, "left".to_owned());
        let i2 = tree.insert(r2, "right".to_owned());
        let i3 = tree.insert(r3, "more right".to_owned());

        tree.erase(i3);
        let i3 = tree.insert(r3, "more right".to_owned());

        let i4 = tree.insert(r4, "more left".to_owned());

        tree.erase(i3);

        for _ in tree.iter_overlap(Rect::new(2.1, 3.0, 0.0, 1.0)) {
            return Err("Erased nodes should not be checked for overlap");
        }

        if !tree.nodes[tree.root].bounds.overlaps(r1)
            || !tree.nodes[tree.root].bounds.overlaps(r2)
            || !tree.nodes[tree.root].bounds.overlaps(r4)
        {
            return Err("Root bounds should still overlap other children");
        }

        if tree.get(i1) != "left" || tree.get(i2) != "right" || tree.get(i4) != "more left" {
            return Err("Other data should remain accessible");
        }

        tree.erase(i1);

        let hit: Vec<&String> = tree
            .iter_overlap(Rect::new(-0.8, 0.2, -0.5, 0.5))
            .map(|s| s)
            .collect();
        if hit != vec!["more left"] {
            return Err("Must overlap only the remaining boxes");
        }

        Ok(())
    }
}
