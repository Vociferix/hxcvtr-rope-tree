mod cursor;
mod mut_cursor;

pub use cursor::Cursor;
pub use mut_cursor::MutCursor;

use std::ops::{Add, Sub};

/// Trait for adapting RopeTree to its contents.
///
/// `Adapter` must be implemented by the user and used as `RopeTree`'s type parameter.
/// `RopeTree` needs to be able to measure the length of its elements, so this trait
/// provides `RopeTree` with that functionality. Rather than require that elements
/// implement a trait, this trait can be implemented on any type, typically an empty
/// type, and is not instantiated by `RopeTree`.
///
/// Along with a function for measuring length, `Adapter` also specifies the element
/// type and size type. The size type, `SizeType`, allows the type used for length and
/// offset into the rope to be customized. In most cases, `SizeType` should be `usize`,
/// but occasionally some other type may be more appropriate. Thus, `Adapter` allows
/// that to be specified. As an example, Hxcvtr specifies `SizeType` to be `u64`
/// because it supports data that are larger than can be addressed by `usize` on 32-bit
/// machines. Additionally, this has the benefit of allowing more exotic use of
/// `RopeTree` where `SizeType` is some more complex user type, as long as said type
/// satisfies the `SizeType` type bounds.
///
/// # Example
///
/// ```
/// extern crate hxcvtr_rope_tree;
/// use hxcvtr_rope_tree::{RopeTree, Adapter};
/// use std::string::String;
///
/// struct MyAdapter;
///
/// impl Adapter for MyAdapter {
///     type Node = String;
///     type SizeType = usize;
///     fn len(node: &String) -> usize {
///         node.len()
///     }
/// }
///
/// type MyRopeTree = RopeTree<MyAdapter>;
/// ```
pub trait Adapter {
    type Node;
    type SizeType: Add<Output=Self::SizeType> + Sub<Output=Self::SizeType> + PartialOrd + Default + Copy;
    fn len(node: &Self::Node) -> Self::SizeType;
}

/// A tree for implementing rope-like data structures.
///
/// `RopeTree` is an ordered, self balancing binary tree, implemented as an AVL Tree.
/// Rather than ordering nodes based on some key that implements `PartialOrd`, nodes
/// can be inserted in arbitrary order, and their order is maintained without a key.
/// Instead of a key, nodes exist at some offset, like in an array. The tree always
/// starts at offset 0. If a node with length `len` is at position `pos`, then the
/// next node, if any, is at position `pos + len`. When new nodes are inserted into
/// the tree, or a node is removed, the position of each node following has its
/// position changed.
///
/// `RopeTree` requires a type that implements `Adapter` as its type parameter. This
/// type provides `RopeTree` with the functionality to measure the length of nodes.
/// See the `Adapter` for more information.
///
/// Most access and manipulation of `RopeTree` is done through `Cursor` and
/// `MutCursor` types. See their documentation for more information.
///
/// # Example
///
/// ```
/// extern crate hxcvtr_rope_tree;
/// use hxcvtr_rope_tree::{RopeTree, Adapter};
/// use std::string::String;
///
/// struct MyAdapter;
///
/// impl Adapter for MyAdapter {
///     type Node = String;
///     type SizeType = usize;
///     fn len(node: &String) -> usize {
///         node.len()
///     }
/// }
///
/// type MyRopeTree = RopeTree<MyAdapter>;
///
/// fn main() {
///     let mut tree: MyRopeTree = RopeTree::with_root(String::from("I am a "));
///     tree.front_mut().insert_after(String::from("Rustacean!"));
///
///     {
///         let mut string = String::new();
///         let mut cursor = tree.front();
///
///         while !cursor.is_null() {
///             string.push_str(cursor.get().unwrap());
///             cursor.move_next();
///         }
///
///         assert_eq!(string.as_str(), "I am a Rustacean!");
///     }
///
///     tree.front_mut().insert_after(String::from("proud "));
///
///     {
///         let mut string = String::new();
///         let mut cursor = tree.front();
///
///         while !cursor.is_null() {
///             string.push_str(cursor.get().unwrap());
///             cursor.move_next();
///         }
///
///         assert_eq!(string.as_str(), "I am a proud Rustacean!");
///     }
/// }
/// ```
pub struct RopeTree<T: Adapter> {
    arena: Vec<Option<Node<T>>>,
    free: Vec<usize>,
    root: usize,
}

struct Node<T: Adapter> {
    parent: usize,
    left: usize,
    right: usize,
    prev: usize,
    next: usize,
    depth: usize,
    weight: T::SizeType,
    data: T::Node,
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
    /// Creates a new empty tree.
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            free: Vec::new(),
            root: NULL,
        }
    }

    /// Creates a new tree with a single node.
    pub fn with_root(root: T::Node) -> Self {
        Self {
            arena: vec![Some(Node::new(NULL, NULL, NULL, root))],
            free: Vec::new(),
            root: 0,
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

    /// Returns the total size of the tree; the sum of the lengths of each node.
    pub fn len(&self) -> T::SizeType {
        match self.try_get(self.root) {
            Some(node) => node.weight,
            None => T::SizeType::default(),
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

    fn set_next(&mut self, node_id: usize, next: usize) {
        self.get_mut(node_id).next = next;
    }

    fn depth(&self, node_id: usize) -> usize {
        self.get(node_id).depth
    }

    fn weight(&self, node_id: usize) -> T::SizeType {
        self.get(node_id).weight
    }

    fn rotate_left(&mut self, node_id: usize) {
        let (parent_id, left_id, right_id) = self.map(node_id, |node| {
            debug_assert_ne!(node.right, NULL);
            (node.parent, node.left, node.right)
        });
        let (left_depth, left_weight) = self.try_map(left_id, |node| {
            (node.depth, node.weight)
        }).unwrap_or((0, T::SizeType::default()));
        let (right_left_id, right_right_id) = self.map(right_id, |node| {
            (node.left, node.right)
        });
        let (right_right_depth, right_right_weight) = self.try_map(right_right_id, |node| {
            (node.depth, node.weight)
        }).unwrap_or((0, T::SizeType::default()));
        let (right_left_depth, right_left_weight) = self.try_map_mut(right_left_id, |node| {
            node.parent = node_id;
            (node.depth, node.weight)
        }).unwrap_or((0, T::SizeType::default()));
        let (depth, weight) = self.map_mut(node_id, |node| {
            node.parent = right_id;
            node.right = right_left_id;
            node.depth = if left_depth > right_left_depth { left_depth } else { right_left_depth } + 1;
            node.weight = left_weight + right_left_weight + T::len(&node.data);
            (node.depth, node.weight)
        });
        self.map_mut(right_id, |node| {
            node.parent = parent_id;
            node.left = node_id;
            node.depth = if depth > right_right_depth { depth } else { right_right_depth } + 1;
            node.weight = weight + right_right_weight + T::len(&node.data);
        });
        if parent_id == NULL {
            self.root = right_id;
        } else {
            self.map_mut(parent_id, |node| {
                let left_id = node.left;
                if left_id == node_id {
                    node.left = right_id;
                } else {
                    node.right = right_id;
                }
            });
        }
    }

    fn rotate_right(&mut self, node_id: usize) {
        let (parent_id, right_id, left_id) = self.map(node_id, |node| {
            debug_assert_ne!(node.left, NULL);
            (node.parent, node.right, node.left)
        });
        let (right_depth, right_weight) = self.try_map(right_id, |node| {
            (node.depth, node.weight)
        }).unwrap_or((0, T::SizeType::default()));
        let (left_right_id, left_left_id) = self.map(left_id, |node| {
            (node.right, node.left)
        });
        let (left_left_depth, left_left_weight) = self.try_map(left_left_id, |node| {
            (node.depth, node.weight)
        }).unwrap_or((0, T::SizeType::default()));
        let (left_right_depth, left_right_weight) = self.try_map_mut(left_right_id, |node| {
            node.parent = node_id;
            (node.depth, node.weight)
        }).unwrap_or((0, T::SizeType::default()));
        let (depth, weight) = self.map_mut(node_id, |node| {
            node.parent = left_id;
            node.left = left_right_id;
            node.depth = if right_depth > left_right_depth { right_depth } else { left_right_depth } + 1;
            node.weight = right_weight + left_right_weight + T::len(&node.data);
            (node.depth, node.weight)
        });
        self.map_mut(left_id, |node| {
            node.parent = parent_id;
            node.right = node_id;
            node.depth = if depth > left_left_depth { depth } else { left_left_depth } + 1;
            node.weight = weight + left_left_weight + T::len(&node.data);
        });
        if parent_id == NULL {
            self.root = left_id;
        } else {
            self.map_mut(parent_id, |node| {
                let right_id = node.right;
                if right_id == node_id {
                    node.right = left_id;
                } else {
                    node.left = left_id;
                }
            });
        }
    }

    /// Returns true if there are no nodes in the tree, false otherwise.
    pub fn is_empty(&self) -> bool {
        self.root == NULL
    }

    /// Removes all nodes from the tree.
    pub fn clear(&mut self) {
        self.root = NULL;
        self.free.clear();
        self.arena.clear();
    }

    /// Returns a null `Cursor`
    pub fn null_cursor(&self) -> Cursor<T> {
        cursor::new(self, T::SizeType::default(), NULL)
    }

    /// Returns a null `MutCursor`
    pub fn null_cursor_mut(&mut self) -> MutCursor<T> {
        mut_cursor::new(self, T::SizeType::default(), NULL)
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

    /// Returns a `Cursor` that points to the front node, at offset 0.
    pub fn front(&self) -> Cursor<T> {
        let front_id = self.front_impl();
        cursor::new(self, T::SizeType::default(), front_id)
    }

    /// Returns a `MutCursor` that points to the front node, at offset 0.
    pub fn front_mut(&mut self) -> MutCursor<T> {
        let front_id = self.front_impl();
        mut_cursor::new(self, T::SizeType::default(), front_id)
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

    /// Returns a `Cursor` that points to the back node.
    pub fn back<'a>(&'a self) -> Cursor<'a, T> {
        let back_id = self.back_impl();
        self.try_map(back_id, |node| {
            cursor::new(self, self.len() - T::len(&node.data), back_id)
        }).unwrap_or(cursor::new(self, T::SizeType::default(), NULL))
    }

    /// Returns a `MutCursor` that points to the back node.
    pub fn back_mut<'a>(&'a mut self) -> MutCursor<'a, T> {
        let back_id = self.back_impl();
        if back_id == NULL {
            mut_cursor::new(self, T::SizeType::default(), NULL)
        } else {
            let back_len = self.map_mut(back_id, |node| {
                T::len(&node.data)
            });
            mut_cursor::new(self, self.len() - back_len, back_id)
        }
    }

    fn upper_bound_impl(&self, pos: T::SizeType) -> (T::SizeType, usize) {
        if self.root == NULL {
            (T::SizeType::default(), NULL)
        } else {
            let mut curr = T::SizeType::default();
            let mut node_id = self.root;
            let mut node = self.get(self.root);
            loop {
                match self.try_get(node.left) {
                    Some(left) => {
                        if curr + left.weight > pos {
                            node_id = node.left;
                            node = left;
                        } else {
                            curr = curr + left.weight;
                            let node_len = T::len(&node.data);
                            if curr + node_len > pos {
                                return (curr, node_id);
                            } else {
                                if node.right == NULL {
                                    return (curr, node_id);
                                } else {
                                    curr = curr + node_len;
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
                            if node.right == NULL {
                                return (curr, node_id);
                            } else {
                                curr = curr + node_len;
                                node_id = node.right;
                                node = self.get(node_id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Returns a `Cursor` to the node with the greatest offset that is less than `pos`.
    /// The cursor will be null if the tree is empty.
    pub fn upper_bound<'a>(&'a self, pos: T::SizeType) -> Cursor<'a, T> {
        let (pos, node_id) = self.upper_bound_impl(pos);
        cursor::new(self, pos, node_id)
    }

    /// Returns a `MutCursor` to the node with the greatest offset that is less than `pos`.
    /// The cursor will be null if the tree is empty.
    pub fn upper_bound_mut<'a>(&'a mut self, pos: T::SizeType) -> MutCursor<'a, T> {
        let (pos, node_id) = self.upper_bound_impl(pos);
        mut_cursor::new(self, pos, node_id)
    }

    fn lower_bound_impl(&self, pos: T::SizeType) -> (T::SizeType, usize) {
        if self.root == NULL {
            (T::SizeType::default(), NULL)
        } else {
            let ret = self.upper_bound_impl(pos);
            if ret.0 < pos {
                let node = self.get(ret.1);
                if node.next == NULL {
                    (self.weight(self.root), NULL)
                } else {
                    (ret.0 + T::len(&node.data), node.next)
                }
            } else {
                ret
            }
        }
    }

    /// Returns a `Cursor` to the node with the least offset that is greater than `pos`.
    /// The cursor will be null if there are no nodes with offset less than `pos`.
    pub fn lower_bound<'a>(&'a self, pos: T::SizeType) -> Cursor<'a, T> {
        let (pos, node_id) = self.lower_bound_impl(pos);
        cursor::new(self, pos, node_id)
    }

    /// Returns a `MutCursor` to the node with the least offset that is greater than `pos`.
    /// The cursor will be null if there are no nodes with offset less than `pos`.
    pub fn lower_bound_mut<'a>(&'a mut self, pos: T::SizeType) -> MutCursor<'a, T> {
        let (pos, node_id) = self.lower_bound_impl(pos);
        mut_cursor::new(self, pos, node_id)
    }

    fn find_impl(&self, pos: T::SizeType) -> (T::SizeType, usize) {
        let len = self.len();
        if pos >= len {
            (len, NULL)
        } else {
            self.upper_bound_impl(pos)
        }
    }

    /// Finds the node that covers the offset `pos`. Unlike `upper_bound`, the returned
    /// cursor will be null if `pos` is greater than or equal to the total tree length.
    /// A `Cursor` and an offset into the node is returned. The offset into the node is
    /// the offset from the start of the node that is equivalent to `pos`. If the cursor
    /// is null, the offset will always be 0, and is essentially meaningless.
    pub fn find<'a>(&'a self, pos: T::SizeType) -> (Cursor<'a, T>, T::SizeType) {
        let (start, node_id) = self.find_impl(pos);
        (cursor::new(self, start, node_id), if node_id == NULL { T::SizeType::default() } else { pos - start })
    }

    /// Finds the node that covers the offset `pos`. Unlike `upper_bound_mut`, the returned
    /// cursor will be null if `pos` is greater than or equal to the total tree length.
    /// A `MutCursor` and an offset into the node is returned. The offset into the node is
    /// the offset from the start of the node that is equivalent to `pos`. If the cursor
    /// is null, the offset will always be 0, and is essentially meaningless.
    pub fn find_mut<'a>(&'a mut self, pos: T::SizeType) -> (MutCursor<'a, T>, T::SizeType) {
        let (start, node_id) = self.find_impl(pos);
        (mut_cursor::new(self, pos, node_id), if node_id == NULL { T::SizeType::default() } else { pos - start })
    }

    fn repair_weight_only(&mut self, mut node_id: usize) {
        while node_id != NULL {
            let (left_id, right_id, parent_id) = self.map(node_id, |node| {
                (node.left, node.right, node.parent)
            });
            let left_weight = self.try_map(left_id, |node| {
                node.weight
            }).unwrap_or(T::SizeType::default());
            let right_weight = self.try_map(right_id, |node| {
                node.weight
            }).unwrap_or(T::SizeType::default());
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
            }).unwrap_or((0, T::SizeType::default()));
            let (right_depth, right_weight) = self.try_map(right_id, |node| {
                (node.depth, node.weight)
            }).unwrap_or((0, T::SizeType::default()));
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
                if balance < 0 {
                    self.rotate_right(right_id);
                    self.rotate_left(node_id);
                } else {
                    self.rotate_left(node_id);
                }
            }
            node_id = parent_id;
        }
    }

    // Unsafe code is used here to more efficiently swap data elements. Multiple
    // mutable into the same tree temporarily exist simultaneously in order to
    // pass them into `std::mem::swap`. If `a_id != b_id` memory safety is
    // guaranteed. The assert, which should never be violated, guarantees that
    // `a_id != b_id`.
    fn data_swap(&mut self, a_id: usize, b_id: usize) {
        assert_ne!(a_id, b_id);
        let a_ptr = (&mut self.get_mut(a_id).data) as *mut T::Node;
        unsafe { std::mem::swap(&mut self.get_mut(b_id).data, &mut *a_ptr) };
    }

    /// Adds a node to the front of the tree.
    ///
    /// Logically equivalent to:
    /// ```
    /// extern crate hxcvtr_rope_tree;
    /// use hxcvtr_rope_tree::{RopeTree, Adapter};
    ///
    /// fn push_front<T: Adapter>(tree: &mut RopeTree<T>, node: T::Node) {
    ///     let mut cursor = tree.front_mut();
    ///     cursor.insert_before(node);
    /// }
    /// ```
    pub fn push_front(&mut self, node: T::Node) {
        let front_id = self.front_impl();
        let node_id = self.alloc(Node::new(front_id, NULL, front_id, node));
        if front_id == NULL {
            self.root = node_id;
        } else {
            self.set_left(front_id, node_id);
            self.repair(front_id);
        }
    }

    /// Adds a node to the back of the tree.
    ///
    /// Logically equivalent to:
    /// ```
    /// extern crate hxcvtr_rope_tree;
    /// use hxcvtr_rope_tree::{RopeTree, Adapter};
    ///
    /// fn push_back<T: Adapter>(tree: &mut RopeTree<T>, node: T::Node) {
    ///     let mut cursor = tree.back_mut();
    ///     cursor.insert_after(node);
    /// }
    /// ```
    pub fn push_back(&mut self, node: T::Node) {
        let back_id = self.back_impl();
        let node_id = self.alloc(Node::new(back_id, back_id, NULL, node));
        if back_id == NULL {
            self.root = node_id;
        } else {
            self.set_right(back_id, node_id);
            self.repair(back_id);
        }
    }

    /// Removes a node from the front of the tree.
    ///
    /// Logically equivalent to:
    /// ```
    /// extern crate hxcvtr_rope_tree;
    /// use hxcvtr_rope_tree::{RopeTree, Adapter};
    ///
    /// fn pop_front<T: Adapter>(tree: &mut RopeTree<T>, node: T::Node) -> Option<T::Node> {
    ///     let mut cursor = tree.front_mut();
    ///     cursor.remove()
    /// }
    /// ```
    pub fn pop_front(&mut self) -> Option<T::Node> {
        let front_id = self.front_impl();
        if front_id == NULL {
            None
        } else {
            let parent_id = self.parent(front_id);
            self.try_map_mut(parent_id, |node| {
                node.left = NULL;
            });
            self.repair(parent_id);
            Some(self.dealloc(front_id).data)
        }
    }


    /// Removes a node from the back of the tree.
    ///
    /// Logically equivalent to:
    /// ```
    /// extern crate hxcvtr_rope_tree;
    /// use hxcvtr_rope_tree::{RopeTree, Adapter};
    ///
    /// fn pop_back<T: Adapter>(tree: &mut RopeTree<T>, node: T::Node) -> Option<T::Node> {
    ///     let mut cursor = tree.back_mut();
    ///     cursor.remove()
    /// }
    /// ```
    pub fn pop_back(&mut self) -> Option<T::Node> {
        let back_id = self.back_impl();
        if back_id == NULL {
            None
        } else {
            let parent_id = self.parent(back_id);
            self.try_map_mut(parent_id, |node| {
                node.right = NULL;
            });
            self.repair(parent_id);
            Some(self.dealloc(back_id).data)
        }
    }
}

#[cfg(test)]
mod tests;
