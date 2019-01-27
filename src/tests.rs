use super::*;

struct TestAdapter;

impl Adapter for TestAdapter {
    type Node = u64;
    type SizeType = u64;
    fn len(node: &Self::Node) -> Self::SizeType {
        *node
    }
}

type TestTree = RopeTree<TestAdapter>;

fn new_test_tree() -> TestTree {
    let mut tree: TestTree = RopeTree::with_root(10);
    {
        let mut cursor = tree.front_mut();
        cursor.insert_after(20);
        cursor.move_next();
        cursor.insert_before(30);
        cursor.move_prev();
        cursor.insert_after(25);
        cursor.move_next();
        cursor.insert_before(15);
        cursor.move_prev();
        cursor.move_prev();
        cursor.insert_after(35);
        cursor.insert_before(40);
    }
    tree
}

#[test]
fn insert_test() {
    let mut tree: TestTree = RopeTree::with_root(10);
    {
        let mut cursor = tree.front_mut();
        assert_eq!(cursor.position().unwrap(), 0);
        assert_eq!(cursor.len().unwrap(), 10);
        assert_eq!(*cursor.get().unwrap(), 10);
        assert_eq!(cursor.tree().len(), 10);

        cursor.insert_after(20);
        assert_eq!(cursor.position().unwrap(), 0);
        assert_eq!(cursor.len().unwrap(), 10);
        assert_eq!(*cursor.get().unwrap(), 10);
        assert_eq!(cursor.tree().len(), 30);

        cursor.move_next();
        assert_eq!(cursor.position().unwrap(), 10);
        assert_eq!(cursor.len().unwrap(), 20);
        assert_eq!(*cursor.get().unwrap(), 20);
        assert_eq!(cursor.tree().len(), 30);

        cursor.insert_before(30);
        assert_eq!(cursor.position().unwrap(), 40);
        assert_eq!(cursor.len().unwrap(), 20);
        assert_eq!(*cursor.get().unwrap(), 20);
        assert_eq!(cursor.tree().len(), 60);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 10);
        assert_eq!(cursor.len().unwrap(), 30);
        assert_eq!(*cursor.get().unwrap(), 30);
        assert_eq!(cursor.tree().len(), 60);

        cursor.insert_after(25);
        assert_eq!(cursor.position().unwrap(), 10);
        assert_eq!(cursor.len().unwrap(), 30);
        assert_eq!(*cursor.get().unwrap(), 30);
        assert_eq!(cursor.tree().len(), 85);

        cursor.move_next();
        assert_eq!(cursor.position().unwrap(), 40);
        assert_eq!(cursor.len().unwrap(), 25);
        assert_eq!(*cursor.get().unwrap(), 25);
        assert_eq!(cursor.tree().len(), 85);

        cursor.insert_before(15);
        assert_eq!(cursor.position().unwrap(), 55);
        assert_eq!(cursor.len().unwrap(), 25);
        assert_eq!(*cursor.get().unwrap(), 25);
        assert_eq!(cursor.tree().len(), 100);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 40);
        assert_eq!(cursor.len().unwrap(), 15);
        assert_eq!(*cursor.get().unwrap(), 15);
        assert_eq!(cursor.tree().len(), 100);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 10);
        assert_eq!(cursor.len().unwrap(), 30);
        assert_eq!(*cursor.get().unwrap(), 30);
        assert_eq!(cursor.tree().len(), 100);

        cursor.insert_after(35);
        assert_eq!(cursor.position().unwrap(), 10);
        assert_eq!(cursor.len().unwrap(), 30);
        assert_eq!(*cursor.get().unwrap(), 30);
        assert_eq!(cursor.tree().len(), 135);

        cursor.move_next();
        assert_eq!(cursor.position().unwrap(), 40);
        assert_eq!(cursor.len().unwrap(), 35);
        assert_eq!(*cursor.get().unwrap(), 35);
        assert_eq!(cursor.tree().len(), 135);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 10);
        assert_eq!(cursor.len().unwrap(), 30);
        assert_eq!(*cursor.get().unwrap(), 30);
        assert_eq!(cursor.tree().len(), 135);

        cursor.insert_before(40);
        assert_eq!(cursor.position().unwrap(), 50);
        assert_eq!(cursor.len().unwrap(), 30);
        assert_eq!(*cursor.get().unwrap(), 30);
        assert_eq!(cursor.tree().len(), 175);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 10);
        assert_eq!(cursor.len().unwrap(), 40);
        assert_eq!(*cursor.get().unwrap(), 40);
        assert_eq!(cursor.tree().len(), 175);
    }

    {
        let mut cursor = tree.front();
        assert_eq!(cursor.position().unwrap(), 0);
        assert_eq!(cursor.len().unwrap(), 10);
        assert_eq!(*cursor.get().unwrap(), 10);

        cursor.move_next();
        assert_eq!(cursor.position().unwrap(), 10);
        assert_eq!(cursor.len().unwrap(), 40);
        assert_eq!(*cursor.get().unwrap(), 40);

        cursor.move_next();
        assert_eq!(cursor.position().unwrap(), 50);
        assert_eq!(cursor.len().unwrap(), 30);
        assert_eq!(*cursor.get().unwrap(), 30);

        cursor.move_next();
        assert_eq!(cursor.position().unwrap(), 80);
        assert_eq!(cursor.len().unwrap(), 35);
        assert_eq!(*cursor.get().unwrap(), 35);

        cursor.move_next();
        assert_eq!(cursor.position().unwrap(), 115);
        assert_eq!(cursor.len().unwrap(), 15);
        assert_eq!(*cursor.get().unwrap(), 15);

        cursor.move_next();
        assert_eq!(cursor.position().unwrap(), 130);
        assert_eq!(cursor.len().unwrap(), 25);
        assert_eq!(*cursor.get().unwrap(), 25);

        cursor.move_next();
        assert_eq!(cursor.position().unwrap(), 155);
        assert_eq!(cursor.len().unwrap(), 20);
        assert_eq!(*cursor.get().unwrap(), 20);
    }

    {
        let mut cursor = tree.back();
        assert_eq!(cursor.position().unwrap(), 155);
        assert_eq!(cursor.len().unwrap(), 20);
        assert_eq!(*cursor.get().unwrap(), 20);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 130);
        assert_eq!(cursor.len().unwrap(), 25);
        assert_eq!(*cursor.get().unwrap(), 25);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 115);
        assert_eq!(cursor.len().unwrap(), 15);
        assert_eq!(*cursor.get().unwrap(), 15);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 80);
        assert_eq!(cursor.len().unwrap(), 35);
        assert_eq!(*cursor.get().unwrap(), 35);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 50);
        assert_eq!(cursor.len().unwrap(), 30);
        assert_eq!(*cursor.get().unwrap(), 30);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 10);
        assert_eq!(cursor.len().unwrap(), 40);
        assert_eq!(*cursor.get().unwrap(), 40);

        cursor.move_prev();
        assert_eq!(cursor.position().unwrap(), 0);
        assert_eq!(cursor.len().unwrap(), 10);
        assert_eq!(*cursor.get().unwrap(), 10);
    }
}

#[test]
fn lower_bound_test_1() {
    let tree = new_test_tree();
    {
        let cursor = tree.lower_bound(60);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 80);
        assert_eq!(cursor.len().unwrap(), 35);
    }
}

#[test]
fn lower_bound_test_2() {
    let tree = new_test_tree();
    {
        let cursor = tree.lower_bound(80);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 80);
        assert_eq!(cursor.len().unwrap(), 35);
    }
}

#[test]
fn lower_bound_test_3() {
    let tree = new_test_tree();
    {
        let cursor = tree.lower_bound(81);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 115);
        assert_eq!(cursor.len().unwrap(), 15);
    }
}

#[test]
fn lower_bound_test_4() {
    let tree = new_test_tree();
    {
        let cursor = tree.lower_bound(0);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 0);
        assert_eq!(cursor.len().unwrap(), 10);
    }
}

#[test]
fn lower_bound_test_5() {
    let tree = new_test_tree();
    {
        let cursor = tree.lower_bound(155);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 155);
        assert_eq!(cursor.len().unwrap(), 20);
    }
}

#[test]
fn lower_bound_test_6() {
    let tree = new_test_tree();
    {
        let cursor = tree.lower_bound(156);
        assert!(cursor.is_null());
        assert!(cursor.position().is_none());
        assert!(cursor.len().is_none());
    }
}

#[test]
fn upper_bound_test_1() {
    let tree = new_test_tree();
    {
        let cursor = tree.upper_bound(85);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 80);
        assert_eq!(cursor.len().unwrap(), 35);
    }
}

#[test]
fn upper_bound_test_2() {
    let tree = new_test_tree();
    {
        let cursor = tree.upper_bound(80);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 80);
        assert_eq!(cursor.len().unwrap(), 35);
    }
}

#[test]
fn upper_bound_test_3() {
    let tree = new_test_tree();
    {
        let cursor = tree.upper_bound(79);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 50);
        assert_eq!(cursor.len().unwrap(), 30);
    }
}

#[test]
fn upper_bound_test_4() {
    let tree = new_test_tree();
    {
        let cursor = tree.upper_bound(0);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 0);
        assert_eq!(cursor.len().unwrap(), 10);
    }
}

#[test]
fn upper_bound_test_5() {
    let tree = new_test_tree();
    {
        let cursor = tree.upper_bound(200);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 155);
        assert_eq!(cursor.len().unwrap(), 20);
    }
}

#[test]
fn find_test_1() {
    let tree = new_test_tree();
    {
        let (cursor, offset) = tree.find(85);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 80);
        assert_eq!(cursor.len().unwrap(), 35);
        assert_eq!(offset, 5);
    }
}

#[test]
fn find_test_2() {
    let tree = new_test_tree();
    {
        let (cursor, offset) = tree.find(80);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 80);
        assert_eq!(cursor.len().unwrap(), 35);
        assert_eq!(offset, 0);
    }
}

#[test]
fn find_test_3() {
    let tree = new_test_tree();
    {
        let (cursor, offset) = tree.find(79);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 50);
        assert_eq!(cursor.len().unwrap(), 30);
        assert_eq!(offset, 29);
    }
}

#[test]
fn find_test_4() {
    let tree = new_test_tree();
    {
        let (cursor, offset) = tree.find(0);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 0);
        assert_eq!(cursor.len().unwrap(), 10);
        assert_eq!(offset, 0);
    }
}

#[test]
fn find_test_5() {
    let tree = new_test_tree();
    {
        let (cursor, offset) = tree.find(174);
        assert!(!cursor.is_null());
        assert_eq!(cursor.position().unwrap(), 155);
        assert_eq!(cursor.len().unwrap(), 20);
        assert_eq!(offset, 19);
    }
}

#[test]
fn find_test_6() {
    let tree = new_test_tree();
    {
        let (cursor, offset) = tree.find(175);
        assert!(cursor.is_null());
        assert_eq!(offset, 0);
    }
}

#[test]
fn remove_test_1() {
    let mut tree = new_test_tree();
    let mut cursor = tree.upper_bound_mut(0);
    let node = cursor.remove();
    assert!(node.is_some());
    assert_eq!(node.unwrap(), 10);
    assert_eq!(cursor.tree().len(), 165);
    assert_eq!(cursor.position().unwrap(), 0);
    assert_eq!(cursor.len().unwrap(), 40);
}

#[test]
fn remove_test_2() {
    let mut tree = new_test_tree();
    let mut cursor = tree.upper_bound_mut(10);
    let node = cursor.remove();
    assert!(node.is_some());
    assert_eq!(node.unwrap(), 40);
    assert_eq!(cursor.tree().len(), 135);
    assert_eq!(cursor.position().unwrap(), 10);
    assert_eq!(cursor.len().unwrap(), 30);
}

#[test]
fn remove_test_3() {
    let mut tree = new_test_tree();
    let mut cursor = tree.upper_bound_mut(50);
    let node = cursor.remove();
    assert!(node.is_some());
    assert_eq!(node.unwrap(), 30);
    assert_eq!(cursor.tree().len(), 145);
    assert_eq!(cursor.position().unwrap(), 50);
    assert_eq!(cursor.len().unwrap(), 35);
}

#[test]
fn remove_test_4() {
    let mut tree = new_test_tree();
    let mut cursor = tree.upper_bound_mut(80);
    let node = cursor.remove();
    assert!(node.is_some());
    assert_eq!(node.unwrap(), 35);
    assert_eq!(cursor.tree().len(), 140);
    assert_eq!(cursor.position().unwrap(), 80);
    assert_eq!(cursor.len().unwrap(), 15);
}

#[test]
fn remove_test_5() {
    let mut tree = new_test_tree();
    let mut cursor = tree.upper_bound_mut(115);
    let node = cursor.remove();
    assert!(node.is_some());
    assert_eq!(node.unwrap(), 15);
    assert_eq!(cursor.tree().len(), 160);
    assert_eq!(cursor.position().unwrap(), 115);
    assert_eq!(cursor.len().unwrap(), 25);
}

#[test]
fn remove_test_6() {
    let mut tree = new_test_tree();
    let mut cursor = tree.upper_bound_mut(130);
    let node = cursor.remove();
    assert!(node.is_some());
    assert_eq!(node.unwrap(), 25);
    assert_eq!(cursor.tree().len(), 150);
    assert_eq!(cursor.position().unwrap(), 130);
    assert_eq!(cursor.len().unwrap(), 20);
}

#[test]
fn remove_test_7() {
    let mut tree = new_test_tree();
    let mut cursor = tree.upper_bound_mut(155);
    let node = cursor.remove();
    assert!(node.is_some());
    assert_eq!(node.unwrap(), 20);
    assert_eq!(cursor.tree().len(), 155);
    assert!(cursor.is_null());
    assert!(cursor.position().is_none());
    assert!(cursor.len().is_none());
    assert!(cursor.get().is_none());
}

#[test]
fn remove_test_8() {
    let mut tree = new_test_tree();
    let mut cursor = tree.null_cursor_mut();
    assert!(cursor.remove().is_none());
}
