pub trait Adapter {
    type Node;
    fn len(node: &Self::Node) -> u64;
}

struct Node<T: Adapter> {
    parent: usize,
    left: usize,
    right: usize,
    prev: usize,
    next: usize,
    depth: usize,
    weight: u64,
    data: T::Node,
}

pub struct RopeTree<T: Adapter> {
    arena: Vec<Option<Node<T>>>,
    free: Vec<usize>,
    root: usize,
}

pub struct Cursor<'a, T: Adapter> {
    tree: &'a RopeTree<T>,
    pos: u64,
    node: usize,
}

pub struct MutCursor<'a, T: Adapter> {
    tree: &'a mut RopeTree<T>,
    pos: u64,
    node: usize,
}

const NULL: usize = std::usize::MAX;

impl<T: Adapter> Node<T> {
    fn new(parent: usize, prev: usize, next: usize, data: T::Node) -> Self {
        Self {
            parent,
            left: NULL,
            right: NULL,
            prev,
            next,
            depth: 1,
            weight: T::len(&data),
            data,
        }
    }
}

impl<T: Adapter> RopeTree<T> {
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            free: Vec::new(),
            root: NULL,
        }
    }

    fn alloc(&mut self, node: Node<T>) -> usize {
        let node_id;
        let fc = self.free.len();
        if fc > 0 {
            node_id = self.free.pop().unwrap();
            self.arena[node_id] = Some(node);
        } else {
            node_id = self.arena.len();
            self.arena.push(Some(node));
        }
        node_id
    }

    fn dealloc(&mut self, node_id: usize) -> Node<T> {
        self.free.push(node_id);
        std::mem::replace(&mut self.arena[node_id], None).unwrap()
    }

    fn try_get(&self, node_id: usize) -> Option<&Node<T>> {
        if node_id < self.arena.len() {
            match self.arena[node_id] {
                Some(ref node) => Some(node),
                None => None,
            }
        } else {
            None
        }
    }

    fn get(&self, node_id: usize) -> &Node<T> {
        self.try_get(node_id).expect("Access on invalid node")
    }

    fn try_get_mut(&mut self, node_id: usize) -> Option<&mut Node<T>> {
        let len = self.arena.len();
        if node_id < len {
            match self.arena[node_id] {
                Some(ref mut node) => Some(node),
                None => None,
            }
        } else {
            None
        }
    }

    fn get_mut(&mut self, node_id: usize) -> &mut Node<T> {
        self.try_get_mut(node_id).expect("Access on invalid node")
    }

    fn try_map<Ret, F: Fn(&Node<T>) -> Ret>(&self, node_id: usize, f: F) -> Option<Ret> {
        match self.try_get(node_id) {
            Some(node) => Some(f(node)),
            None => None,
        }
    }

    fn map<Ret, F: Fn(&Node<T>) -> Ret>(&self, node_id: usize, f: F) -> Ret {
        self.try_map(node_id, f).expect("Access on invalid node")
    }

    fn try_map_mut<Ret, F: Fn(&mut Node<T>) -> Ret>(&mut self, node_id: usize, f: F) -> Option<Ret> {
        match self.try_get_mut(node_id) {
            Some(node) => Some(f(node)),
            None => None,
        }
    }

    fn map_mut<Ret, F: Fn(&mut Node<T>) -> Ret>(&mut self, node_id: usize, f: F) -> Ret {
        self.try_map_mut(node_id, f).expect("Access on invalid node")
    }

    pub fn len(&self) -> u64 {
        match self.try_get(self.root) {
            Some(node) => node.weight,
            None => 0,
        }
    }

    fn parent(&self, node_id: usize) -> usize {
        self.get(node_id).parent
    }

    fn set_parent(&mut self, node_id: usize, parent: usize) {
        self.get_mut(node_id).parent = parent;
    }

    fn left(&self, node_id: usize) -> usize {
        self.get(node_id).left
    }

    fn set_left(&mut self, node_id: usize, left: usize) {
        self.get_mut(node_id).left = left;
    }

    fn right(&self, node_id: usize) -> usize {
        self.get(node_id).right
    }

    fn set_right(&mut self, node_id: usize, right: usize) {
        self.get_mut(node_id).right = right;
    }

    fn prev(&self, node_id: usize) -> usize {
        self.get(node_id).prev
    }

    fn set_prev(&mut self, node_id: usize, prev: usize) {
        self.get_mut(node_id).prev = prev;
    }

    fn next(&self, node_id: usize) -> usize {
        self.get(node_id).next
    }

    fn set_next(&mut self, node_id: usize, next: usize) {
        self.get_mut(node_id).next = next;
    }

    fn depth(&self, node_id: usize) -> usize {
        self.get(node_id).depth
    }

    fn set_depth(&mut self, node_id: usize, depth: usize) {
        self.get_mut(node_id).depth = depth;
    }

    fn weight(&self, node_id: usize) -> u64 {
        self.get(node_id).weight
    }

    fn set_weight(&mut self, node_id: usize, weight: u64) {
        self.get_mut(node_id).weight = weight;
    }

    fn rotate_left(&mut self, node_id: usize) -> usize {
        let (parent_id, right_id, left_id) = self.map(node_id, |node| {
            assert_ne!(node.right, NULL);
            (node.parent, node.right, node.left)
        });
        let right_left_id = self.left(right_id);
        let (right_left_depth, right_left_weight) = self.try_map(right_left_id, |node| {
            (node.depth, node.weight)
        }).unwrap_or((0, 0));
        let (left_depth, left_weight) = self.try_map(left_id, |node| {
            (node.depth, node.weight)
        }).unwrap_or((0, 0));

        let (depth, weight) = self.map_mut(node_id, |node| {
            node.parent = right_id;
            node.right = right_left_id;
            node.depth = if left_depth < right_left_depth {
                right_left_depth + 1
            } else {
                left_depth + 1
            };
            node.weight = left_weight + right_left_weight + T::len(&node.data);
            (node.depth, node.weight)
        });

        let (right_right_depth, right_right_weight) = self.try_map(self.right(right_id), |node| {
            (node.depth, node.weight)
        }).unwrap_or((0, 0));
        let (right_depth, right_weight) = self.map_mut(right_id, |node| {
            node.parent = parent_id;
            node.left = node_id;
            node.depth = if depth < right_right_depth {
                right_right_depth + 1
            } else {
                depth + 1
            };
            node.weight = weight + right_right_weight + T::len(&node.data);
            (node.depth, node.weight)
        });

        if parent_id != NULL {
            let parent_other_id = self.map(parent_id, |node| {
                if node.left == node_id {
                    node.right
                } else {
                    node.left
                }
            });
            let (parent_other_depth, parent_other_weight) = self.try_map(parent_other_id, |node| {
                (node.depth, node.weight)
            }).unwrap_or((0, 0));
            self.map_mut(parent_id, |node| {
                node.depth = if parent_other_depth < right_depth {
                    right_depth + 1
                } else {
                    parent_other_depth + 1
                };
                node.weight = parent_other_weight + right_weight + T::len(&node.data);
            });
        } else {
            self.root = right_id;
        }

        right_id
    }

    fn rotate_right(&mut self, node_id: usize) -> usize {
        let (parent_id, right_id, left_id) = self.map(node_id, |node| {
            assert_ne!(node.right, NULL);
            (node.parent, node.right, node.left)
        });
        let left_right_id = self.right(left_id);
        let (left_right_depth, left_right_weight) = self.try_map(left_right_id, |node| {
            (node.depth, node.weight)
        }).unwrap_or((0, 0));
        let (right_depth, right_weight) = self.try_map(right_id, |node| {
            (node.depth, node.weight)
        }).unwrap_or((0, 0));

        let (depth, weight) = self.map_mut(node_id, |node| {
            node.parent = left_id;
            node.right = left_right_id;
            node.depth = if right_depth < left_right_depth {
                left_right_depth + 1
            } else {
                right_depth + 1
            };
            node.weight = right_weight + left_right_weight + T::len(&node.data);
            (node.depth, node.weight)
        });

        let (left_left_depth, left_left_weight) = self.try_map(self.left(left_id), |node| {
            (node.depth, node.weight)
        }).unwrap_or((0, 0));
        let (left_depth, left_weight) = self.map_mut(left_id, |node| {
            node.parent = parent_id;
            node.right = node_id;
            node.depth = if depth < left_left_depth {
                left_left_depth + 1
            } else {
                depth + 1
            };
            node.weight = weight + left_left_weight + T::len(&node.data);
            (node.depth, node.weight)
        });

        if parent_id != NULL {
            let parent_other_id = self.map(parent_id, |node| {
                if node.left == node_id {
                    node.right
                } else {
                    node.left
                }
            });
            let (parent_other_depth, parent_other_weight) = self.try_map(parent_other_id, |node| {
                (node.depth, node.weight)
            }).unwrap_or((0, 0));
            self.map_mut(parent_id, |node| {
                node.depth = if parent_other_depth < left_depth {
                    left_depth + 1
                } else {
                    parent_other_depth + 1
                };
                node.weight = parent_other_weight + left_weight + T::len(&node.data);
            });
        } else {
            self.root = left_id;
        }

        left_id
    }

    pub fn is_empty(&self) -> bool {
        self.root == NULL
    }

    pub fn clear(&mut self) {
        self.root = NULL;
        self.free.clear();
        self.arena.clear();
    }

    fn front_impl(&self) -> usize {
        if self.root == NULL {
            NULL
        } else {
            let mut node_id = self.root;
            loop {
                let left_id = self.left(node_id);
                if left_id == NULL {
                    break node_id;
                } else {
                    node_id = left_id;
                }
            }
        }
    }

    pub fn front(&self) -> Cursor<T> {
        let front_id = self.front_impl();
        Cursor::new(self, 0, front_id)
    }

    pub fn front_mut(&mut self) -> MutCursor<T> {
        let front_id = self.front_impl();
        MutCursor::new(self, 0, front_id)
    }

    fn back_impl(&self) -> usize {
        if self.root == NULL {
            NULL
        } else {
            let mut node_id = self.root;
            loop {
                let right_id = self.right(node_id);
                if right_id == NULL {
                    break node_id;
                } else {
                    node_id = right_id;
                }
            }
        }
    }

    pub fn back<'a>(&'a self) -> Cursor<'a, T> {
        let back_id = self.back_impl();
        self.try_map(back_id, |node| {
            Cursor::new(self, self.len() - T::len(&node.data), back_id)
        }).unwrap_or(Cursor::new(self, 0, NULL))
    }

    pub fn back_mut<'a>(&'a mut self) -> MutCursor<'a, T> {
        let back_id = self.back_impl();
        if back_id == NULL {
            MutCursor::new(self, 0, NULL)
        } else {
            let back_len = self.map_mut(back_id, |node| {
                T::len(&node.data)
            });
            MutCursor::new(self, self.len() - back_len, back_id)
        }
    }

    fn upper_bound_impl(&self, pos: u64) -> (u64, usize) {
        if self.root == NULL {
            (0, NULL)
        } else {
            let mut curr = 0;
            let mut node_id = self.root;
            let mut node = self.get(self.root);
            loop {
                match self.try_get(node.left) {
                    Some(left) => {
                        if curr + left.weight > pos {
                            node_id = node.left;
                            node = left;
                        } else {
                            curr += left.weight;
                            let node_len = T::len(&node.data);
                            if curr + node_len > pos {
                                return (curr, node_id);
                            } else {
                                curr += node_len;
                                if node.right == NULL {
                                    return (curr, node_id);
                                } else {
                                    node_id = node.right;
                                    node = self.get(node_id);
                                }
                            }
                        }
                    },
                    None => {
                        let node_len = T::len(&node.data);
                        if curr + node_len > pos {
                            return (curr, node_id);
                        } else {
                            curr += node_len;
                            if node.right == NULL {
                                return (curr, node_id);
                            } else {
                                node_id = node.right;
                                node = self.get(node_id);
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn upper_bound<'a>(&'a self, pos: u64) -> Cursor<'a, T> {
        let (pos, node_id) = self.upper_bound_impl(pos);
        Cursor::new(self, pos, node_id)
    }

    pub fn upper_bound_mut<'a>(&'a mut self, pos: u64) -> MutCursor<'a, T> {
        let (pos, node_id) = self.upper_bound_impl(pos);
        MutCursor::new(self, pos, node_id)
    }

    fn lower_bound_impl(&self, pos: u64) -> (u64, usize) {
        let (curr, node_id) = self.upper_bound_impl(pos);
        if node_id == NULL {
            (curr, NULL)
        } else {
            let len = self.map(node_id, |node| {
                T::len(&node.data)
            });
            let next_id = self.next(node_id);
            (curr + len, next_id)
        }
    }

    pub fn lower_bound<'a>(&'a self, pos: u64) -> Cursor<'a, T> {
        let (pos, node_id) = self.lower_bound_impl(pos);
        Cursor::new(self, pos, node_id)
    }

    pub fn lower_bound_mut<'a>(&'a mut self, pos: u64) -> MutCursor<'a, T> {
        let (pos, node_id) = self.lower_bound_impl(pos);
        MutCursor::new(self, pos, node_id)
    }

    pub fn insert<'a>(&'a mut self, pos: u64, node: T::Node) -> MutCursor<'a, T> {
        let mut c = self.upper_bound_mut(pos);
        c.insert_after(pos, node);
        c.move_next();
        c
    }

    fn repair_weight_only(&mut self, mut node_id: usize) {
        while node_id != NULL {
            let (left_id, right_id, parent_id) = self.map(node_id, |node| {
                (node.left, node.right, node.parent)
            });
            let left_weight = self.try_map(left_id, |node| {
                node.weight
            }).unwrap_or(0);
            let right_weight = self.try_map(right_id, |node| {
                node.weight
            }).unwrap_or(0);
            self.map_mut(node_id, |node| {
                node.weight = left_weight + right_weight + T::len(&node.data);
            });
            node_id = parent_id;
        }
    }

    fn repair(&mut self, mut node_id: usize) {
        while node_id != NULL {
            let (left_id, right_id, parent_id) = self.map(node_id, |node| {
                (node.left, node.right, node.parent)
            });
            let (left_depth, left_weight) = self.try_map(left_id, |node| {
                (node.depth, node.weight)
            }).unwrap_or((0, 0));
            let (right_depth, right_weight) = self.try_map(right_id, |node| {
                (node.depth, node.weight)
            }).unwrap_or((0, 0));
            self.map_mut(node_id, |node| {
                node.depth = if left_depth < right_depth {
                    right_depth + 1
                } else {
                    left_depth + 1
                };
                node.weight = left_weight + right_weight + T::len(&node.data);
            });

            let balance = right_depth as i64 - left_depth as i64;
            if balance < -1 {
                let (left_left_id, left_right_id) = self.map(left_id, |node| {
                    (node.left, node.right)
                });
                let left_left_depth = if left_left_id == NULL { 0 } else { self.depth(left_left_id) };
                let left_right_depth = if left_right_id == NULL { 0 } else { self.depth(left_right_id) };
                let balance = left_right_depth as i64 - left_left_depth as i64;
                if balance > 0 {
                    self.rotate_left(left_id);
                    self.rotate_right(node_id);
                } else {
                    self.rotate_right(node_id);
                }
            } else if balance > 1 {
                let (right_left_id, right_right_id) = self.map(right_id, |node| {
                    (node.left, node.right)
                });
                let right_left_depth = if right_left_id == NULL { 0 } else { self.depth(right_left_id) };
                let right_right_depth = if right_right_id == NULL { 0 } else { self.depth(right_right_id) };
                let balance = right_right_depth as i64 - right_left_depth as i64;
                if balance > 0 {
                    self.rotate_right(right_id);
                    self.rotate_left(node_id);
                } else {
                    self.rotate_left(node_id);
                }
            }
            node_id = parent_id;
        }
    }
}

impl<'a, T: Adapter> Clone for Cursor<'a, T> {
    fn clone(&self) -> Self {
        Self {
            tree: self.tree,
            pos: self.pos,
            node: self.node,
        }
    }
}

impl<'a, T: Adapter> Cursor<'a, T> {
    fn new(tree: &'a RopeTree<T>, pos: u64, node: usize) -> Self {
        Self {
            tree,
            pos,
            node,
        }
    }

    pub fn tree(&self) -> &RopeTree<T> {
        self.tree
    }

    pub fn is_null(&self) -> bool {
        self.node == NULL
    }

    pub fn position(&self) -> Option<u64> {
        if self.node == NULL {
            None
        } else {
            Some(self.pos)
        }
    }

    pub fn get(&self) -> Option<&T::Node> {
        self.tree.try_get(self.node).map(|node| &node.data)
    }

    pub fn move_next(&mut self) {
        let node_id = self.node;
        if node_id == NULL {
            self.pos = 0;
            self.node = self.tree.front_impl();
        } else {
            let (next, len) = self.tree.map(self.node, |node| {
                (node.next, T::len(&node.data))
            });
            self.pos += len;
            self.node = next;
        }
    }

    pub fn move_prev(&mut self) {
        let node_id = self.node;
        if node_id == NULL {
            let root = self.tree.root;
            if root == NULL {
                self.pos = 0;
                self.node = NULL;
            } else {
                let back_id = self.tree.back_impl();
                self.pos = self.tree.weight(root) - self.tree.map(back_id, |node| T::len(&node.data));
                self.node = back_id;
            }
        } else {
            let prev_id = self.tree.prev(node_id);
            if prev_id == NULL {
                self.pos = 0;
            } else {
                self.pos -= self.tree.map(prev_id, |node| T::len(&node.data))
            }
        }
    }

    pub fn peek_next(&self) -> Self {
        let mut ret = self.clone();
        ret.move_next();
        ret
    }

    pub fn peek_prev(&self) -> Self {
        let mut ret = self.clone();
        ret.move_prev();
        ret
    }
}

impl<'a, T: Adapter> MutCursor<'a, T> {
    fn new(tree: &'a mut RopeTree<T>, pos: u64, node: usize) -> Self {
        Self {
            tree,
            pos,
            node,
        }
    }

    pub fn tree(&self) -> &RopeTree<T> {
        self.tree
    }

    pub fn tree_mut(&mut self) -> &mut RopeTree<T> {
        self.tree
    }

    pub fn is_null(&self) -> bool {
        self.node == NULL
    }

    pub fn position(&self) -> Option<u64> {
        if self.node == NULL {
            None
        } else {
            Some(self.pos)
        }
    }

    pub fn get(&self) -> Option<&T::Node> {
        self.tree.try_get(self.node).map(|node| &node.data)
    }

    pub fn mutate<F: Fn(&mut T::Node)>(&mut self, f: F) {
        let node_id = self.node;
        if node_id != NULL {
            let len = self.tree.map(node_id, |node| T::len(&node.data));
            self.tree.try_map_mut(self.node, |node| {
                f(&mut node.data);
            });
            let new_len = self.tree.map(node_id, |node| T::len(&node.data));
            if len != new_len {
                self.tree.repair_weight_only(node_id);
            }
        }
    }

    pub fn as_cursor(&'a self) -> Cursor<'a, T> {
        Cursor::new(self.tree, self.pos, self.node)
    }

    pub fn move_next(&mut self) {
        let node_id = self.node;
        if node_id == NULL {
            self.pos = 0;
            self.node = self.tree.front_impl();
        } else {
            let (next, len) = self.tree.map(self.node, |node| {
                (node.next, T::len(&node.data))
            });
            self.pos += len;
            self.node = next;
        }
    }

    pub fn move_prev(&mut self) {
        let node_id = self.node;
        if node_id == NULL {
            let root = self.tree.root;
            if root == NULL {
                self.pos = 0;
                self.node = NULL;
            } else {
                let back_id = self.tree.back_impl();
                self.pos = self.tree.weight(root) - self.tree.map(back_id, |node| T::len(&node.data));
                self.node = back_id;
            }
        } else {
            let prev_id = self.tree.prev(node_id);
            if prev_id == NULL {
                self.pos = 0;
            } else {
                self.pos -= self.tree.map(prev_id, |node| T::len(&node.data))
            }
        }
    }

    pub fn peek_next(&'a self) -> Cursor<'a, T> {
        let mut ret = self.as_cursor();
        ret.move_next();
        ret
    }

    pub fn peek_prev(&'a self) -> Cursor<'a, T> {
        let mut ret = self.as_cursor();
        ret.move_prev();
        ret
    }

    pub fn remove(&mut self) -> Option<T::Node> {
        if self.node == NULL {
            return None;
        }

        let node_id = self.node;
        let (left_id, right_id, parent_id, prev_id, next_id) = self.tree.map(node_id, |node| {
            (node.left, node.right, node.parent, node.prev, node.next)
        });

        if left_id == NULL {
            if right_id == NULL {
                // no leaves
                let res = self.tree.try_map_mut(parent_id, |node| {
                    let left_id = node.left;
                    if left_id == node_id {
                        node.left = NULL;
                        node.prev = prev_id;
                        true
                    } else {
                        node.right = NULL;
                        node.next = next_id;
                        false
                    }
                });
                match res {
                    Some(true) => {
                        if prev_id != NULL {
                            self.tree.set_next(prev_id, parent_id)
                        }
                        self.tree.repair(parent_id)
                    },
                    Some(false) => {
                        if next_id != NULL {
                            self.tree.set_prev(next_id, parent_id)
                        }
                        self.tree.repair(parent_id);
                    },
                    None => self.tree.root = NULL,
                }
            } else {
                // right leaf only
                let res = self.tree.try_map_mut(parent_id, |node| {
                    let left_id = node.left;
                    if left_id == node_id {
                        node.left = right_id;
                    } else {
                        node.right = right_id;
                    }
                });
                if prev_id != NULL {
                    self.tree.set_next(prev_id, next_id);
                }
                if next_id != NULL {
                    self.tree.set_prev(next_id, prev_id);
                }
                self.tree.set_parent(right_id, parent_id);
                if res.is_none() {
                    self.tree.root = right_id;
                    self.tree.repair(parent_id);
                }
            }
        } else if right_id == NULL {
            // left leaf only
            let res = self.tree.try_map_mut(parent_id, |node| {
                let left_id = node.left;
                if left_id == node_id {
                    node.left = left_id;
                } else {
                    node.right = left_id;
                }
            });
            if prev_id != NULL {
                self.tree.set_next(prev_id, next_id);
            }
            if next_id != NULL {
                self.tree.set_prev(next_id, prev_id);
            }
            self.tree.set_parent(left_id, parent_id);
            if res.is_none() {
                self.tree.root = right_id;
                self.tree.repair(parent_id);
            }
        } else {
            // both leaves
            let (next_parent_id, next_left_id, next_right_id, next_prev_id, next_next_id) = self.tree.map(next_id, |node| {
                (node.parent, node.left, node.right, node.prev, node.next, T::len(&node.data))
            });
            self.tree.map_mut(node_id, |node| {
                node.parent = next_parent_id;
                node.left = next_left_id;
                node.right = next_right_id;
                node.prev = next_id;
                node.next = next_next_id;
            });
            self.tree.map_mut(next_id, |node| {
                node.parent = parent_id;
                node.left = left_id;
                node.right = right_id;
                node.prev = prev_id;
                node.next = node_id;
            });
            if parent_id == NULL {
                self.tree.root = next_id;
            } else {
                self.tree.map_mut(parent_id, |node| {
                    let left_id = node.left;
                    if left_id == node_id {
                        node.left = next_id;
                    } else {
                        node.right = next_id;
                    }
                });
            }
            if next_parent_id == NULL {
                self.tree.root = node_id;
            } else {
                self.tree.map_mut(next_parent_id, |node| {
                    let left_id = node.left;
                    if left_id == next_id {
                        node.left = node_id;
                    } else {
                        node.right = node_id;
                    }
                });
            }
            if left_id != NULL {
                self.tree.set_parent(left_id, next_id);
            }
            if next_left_id != NULL {
                self.tree.set_parent(next_left_id, node_id);
            }
            if right_id != NULL {
                self.tree.set_parent(right_id, next_id);
            }
            if next_right_id != NULL {
                self.tree.set_parent(next_right_id, node_id);
            }
            if prev_id != NULL {
                self.tree.set_next(prev_id, next_id);
            }
            if next_next_id != NULL {
                self.tree.set_prev(next_next_id, node_id);
            }
            return self.remove();
        }
        self.node = next_id;
        Some(self.tree.dealloc(node_id).data)
    }

    pub fn replace_with(&mut self, node: T::Node) -> Option<T::Node> {
        let ret = match self.tree.try_get_mut(self.node) {
            Some(n) => Some(std::mem::replace(&mut n.data, node)),
            None => None,
        };
        if ret.is_some() {
            self.tree.repair_weight_only(self.node);
        }
        ret
    }

    pub fn insert_before(&mut self, pos: u64, node: T::Node) {
        let mut node_id = self.node;
        if node_id == NULL {
            self.move_prev();
        }

        let (left_id, prev_id) = self.tree.map(self.node, |node| {
            (node.left, node.prev)
        });
        if left_id == NULL {
            let tmp = self.tree.alloc(Node::new(node_id, prev_id, node_id, node));
            self.tree.set_left(node_id, tmp);
            self.tree.set_prev(node_id, tmp);
            self.tree.set_next(prev_id, tmp);
        } else {
            loop {
                let right_id = self.tree.right(node_id);
                if right_id == NULL {
                    break;
                }
                node_id = right_id;
            }
            let tmp = self.tree.alloc(Node::new(node_id, prev_id, node_id, node));
            self.tree.set_right(node_id, tmp);
            self.tree.set_prev(node_id, tmp);
            self.tree.set_next(prev_id, tmp);
        }

        self.tree.repair(node_id);
    }

    pub fn insert_after(&mut self, pos: u64, node: T::Node) {
        let mut node_id = self.node;
        if node_id == NULL {
            self.move_next();
        }

        let (right_id, next_id) = self.tree.map(self.node, |node| {
            (node.right, node.next)
        });
        if right_id == NULL {
            let tmp = self.tree.alloc(Node::new(node_id, node_id, next_id, node));
            self.tree.set_right(node_id, tmp);
            self.tree.set_next(node_id, tmp);
            self.tree.set_prev(next_id, tmp);
        } else {
            loop {
                let left_id = self.tree.left(node_id);
                if left_id == NULL {
                    break;
                }
                node_id = left_id;
            }
            let tmp = self.tree.alloc(Node::new(node_id, node_id, next_id, node));
            self.tree.set_left(node_id, tmp);
            self.tree.set_next(node_id, tmp);
            self.tree.set_prev(next_id, tmp);
        }

        self.tree.repair(node_id);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
