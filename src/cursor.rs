use super::*;

/// A cursor into a tree that provides immutable access and traversal of a `RopeTree`.
///
/// `Cursor` only provides read access into the tree. The cursor can be moved back
/// and forth between nodes. `RopeTree` does not implement any iterators, but `Cursor`
/// can provide that functionality. Due to access into the tree being immutable,
/// multiple `Cursor`s can coexist safely.
///
/// The `Cursor` may either point to a single node in the tree, or null, meaning that
/// it does not point to any node.
pub struct Cursor<'a, T: Adapter> {
    tree: &'a RopeTree<T>,
    pos: T::SizeType,
    node: usize,
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

pub fn new<T: Adapter>(tree: &RopeTree<T>, pos: T::SizeType, node: usize) -> Cursor<T> {
    Cursor {
        tree,
        pos,
        node,
    }
}

impl<'a, T: Adapter> Cursor<'a, T> {
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
    pub fn peek_next(&self) -> Self {
        let mut ret = self.clone();
        ret.move_next();
        ret
    }

    /// Returns a new cursor to the previous node in the tree. If the
    /// self cursor is on the front node, the new cursor will be null.
    /// If the self cursor is null, the new cursor will be on the back
    /// node.
    pub fn peek_prev(&self) -> Self {
        let mut ret = self.clone();
        ret.move_prev();
        ret
    }
}
