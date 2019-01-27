use super::*;

/// A cursor into a tree that provides mutable access and traveral of a `RopeTree`.
///
/// `MutCursor` provides much of the same functionality as `Cursor`, except that
/// `MutCursor` also provides functions for mutating the tree. Because of this
/// mutability, only one `MutCursor` can be instantiated at one time. Along with
/// providing mutable access to the contents of a node, a `MutCursor` can also be
/// used to insert nodes before or after the cursor's node and to remove the
/// cursor's node.
pub struct MutCursor<'a, T: Adapter> {
    tree: &'a mut RopeTree<T>,
    pos: T::SizeType,
    node: usize,
}

pub fn new<T: Adapter>(tree: &mut RopeTree<T>, pos: T::SizeType, node: usize) -> MutCursor<T> {
    MutCursor {
        tree,
        pos,
        node,
    }
}

impl<'a, T: Adapter> MutCursor<'a, T> {
    /// Returns an immutable reference to the tree the cursor points into.
    pub fn tree(&self) -> &RopeTree<T> {
        self.tree
    }

    /// Returns true if the cursor is null, false otherwise.
    pub fn is_null(&self) -> bool {
        self.node == NULL
    }

    /// Returns the start offset of the node the cursor points to. `None`
    /// if the cursor is null.
    pub fn position(&self) -> Option<T::SizeType> {
        if self.node == NULL {
            None
        } else {
            Some(self.pos)
        }
    }

    /// Returns the length of the node the cursor points to. `None` if the
    /// cursor is null.
    pub fn len(&self) -> Option<T::SizeType> {
        if self.node == NULL {
            None
        } else {
            Some(T::len(&self.tree.get(self.node).data))
        }
    }

    /// Returns an immutable reference to the node element of the node the
    /// cursor points to. `None` if the cursor is null.
    pub fn get(&self) -> Option<&T::Node> {
        self.tree.try_get(self.node).map(|node| &node.data)
    }

    /// Provides mutable access to node data via a closure. Mutable access
    /// must be done through a closure so that the tree can be repaired if
    /// the length of the node is changed.
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

    /// Returns a `Cursor` whose lifetime is tied to the `MutCursor`.
    pub fn as_cursor(&'a self) -> Cursor<'a, T> {
        cursor::new(self.tree, self.pos, self.node)
    }

    /// Moves the cursor to the next node in the tree. If the cursor is on
    /// the back node, the cursor will become null. If the cursor is null,
    /// the cursor will move to the front node.
    pub fn move_next(&mut self) {
        let node_id = self.node;
        if node_id == NULL {
            self.pos = T::SizeType::default();
            self.node = self.tree.front_impl();
        } else {
            let (next, len) = self.tree.map(self.node, |node| {
                (node.next, T::len(&node.data))
            });
            self.pos = self.pos + len;
            self.node = next;
        }
    }

    /// Move the cursor to the previous node in the tree. If the cursor is
    /// on the front node, the cursor will become null. If the cursor is
    /// null, the cursor will move to the back node.
    pub fn move_prev(&mut self) {
        let node_id = self.node;
        if node_id == NULL {
            let root = self.tree.root;
            if root == NULL {
                self.pos = T::SizeType::default();
                self.node = NULL;
            } else {
                let back_id = self.tree.back_impl();
                self.pos = self.tree.weight(root) - self.tree.map(back_id, |node| T::len(&node.data));
                self.node = back_id;
            }
        } else {
            let prev_id = self.tree.prev(node_id);
            if prev_id == NULL {
                self.pos = T::SizeType::default();
            } else {
                self.pos = self.pos - self.tree.map(prev_id, |node| T::len(&node.data))
            }
            self.node = prev_id;
        }
    }

    /// Returns a new cursor to the next node in the tree. If the self
    /// cursor is on the back node, the new cursor will be null. If the
    /// self cursor is null, the new cursor will be on the front node.
    pub fn peek_next(&'a self) -> Cursor<'a, T> {
        let mut ret = self.as_cursor();
        ret.move_next();
        ret
    }

    /// Returns a new cursor to the previous node in the tree. If the
    /// self cursor is on the front node, the new cursor will be null.
    /// If the self cursor is null, the new cursor will be on the back
    /// node.
    pub fn peek_prev(&'a self) -> Cursor<'a, T> {
        let mut ret = self.as_cursor();
        ret.move_prev();
        ret
    }

    /// Removes the node the cursor points to from the tree. The cursor
    /// will move to the next node in the tree. If the cursor is null,
    /// the tree is unchanged and `None` is returned. If the cursor is
    /// is on the back node, the cursor will be null after removal.
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
                } else {
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
            } else {
                self.tree.repair(parent_id);
            }
        } else {
            // both leaves
            self.tree.data_swap(node_id, prev_id);
            self.node = prev_id;
            let ret = self.remove();
            self.node = next_id;
            return ret;
        }
        self.node = next_id;
        Some(self.tree.dealloc(node_id).data)
    }

    /// Replaces the data of the node the cursor points to. The old
    /// data is returned. If the cursor is null, `None` is returned.
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

    /// Insert a node before the node the cursor points to. The cursor
    /// position is unchanged. If the cursor is null, the new node is
    /// inserted at the back of the tree.
    pub fn insert_before(&mut self, node: T::Node) {
        let node_id = self.node;
        if node_id == NULL {
            self.move_prev();
        }
        let len = T::len(&node);

        let (left_id, prev_id) = self.tree.map(self.node, |node| {
            (node.left, node.prev)
        });
        if left_id == NULL {
            let tmp = self.tree.alloc(Node::new(node_id, prev_id, node_id, node));
            self.tree.set_left(node_id, tmp);
            self.tree.set_prev(node_id, tmp);
            self.tree.set_next(prev_id, tmp);
            self.tree.repair(node_id);
        } else {
            let tmp = self.tree.alloc(Node::new(prev_id, prev_id, node_id, node));
            self.tree.set_right(prev_id, tmp);
            self.tree.set_prev(node_id, tmp);
            self.tree.set_next(prev_id, tmp);
            self.tree.repair(prev_id);
        }
        self.pos = self.pos + len;
    }

    /// Insert a node after the node the cursor points to. The cursor
    /// position is unchanged. If the cursor is null, the new node is
    /// inserted at the front of the tree.
    pub fn insert_after(&mut self, node: T::Node) {
        let node_id = self.node;
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
            if next_id != NULL {
                self.tree.set_prev(next_id, tmp);
            }
            self.tree.repair(node_id);
        } else {
            let tmp = self.tree.alloc(Node::new(next_id, node_id, next_id, node));
            self.tree.set_left(next_id, tmp);
            self.tree.set_next(node_id, tmp);
            self.tree.set_prev(next_id, tmp);
            self.tree.repair(next_id);
        }
    }
}
