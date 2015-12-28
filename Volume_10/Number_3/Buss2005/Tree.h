
#include "LinearR3.h"
#include "Node.h"

#ifndef _CLASS_TREE
#define _CLASS_TREE

class Tree {

public:
	Tree();

	int GetNumNode() const { return nNode; }
	int GetNumEffector() const { return nEffector; }
	int GetNumJoint() const { return nJoint; }
	void InsertRoot(Node*);
	void InsertLeftChild(Node* parent, Node* child);
	void InsertRightSibling(Node* parent, Node* child);

	// Accessors based on node numbers
	Node* GetJoint(int);
	Node* GetEffector(int);
	const VectorR3& GetEffectorPosition(int);

	// Accessors for tree traversal
	Node* GetRoot() const { return root; }
	Node* GetSuccessor ( const Node* ) const;
	Node* GetParent( const Node* node ) const { return node->realparent; }

	void Compute();
	void Draw();
	void Print();
	void Init();
	void UnFreeze();

private:
	Node* root;
	int nNode;			// nNode = nEffector + nJoint
	int nEffector;
	int nJoint;
	void SetSeqNum(Node*);
	Node* SearchJoint(Node*, int);
	Node* SearchEffector(Node*, int);
	void ComputeTree(Node*);
	void DrawTree(Node*);
	void PrintTree(Node*);
	void InitTree(Node*);
	void UnFreezeTree(Node*);
};

inline Node* Tree::GetSuccessor ( const Node* node ) const
{
	if ( node->left ) {
		return node->left;
	}
	while ( true ) {
		if ( node->right ) {
			return ( node->right );
		}
		node = node->realparent;
		if ( !node ) {
			return 0;		// Back to root, finished traversal
		} 
	}
}

#endif