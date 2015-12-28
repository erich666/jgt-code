
#include <iostream>
using namespace std;

#ifdef WIN32
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "LinearR3.h"
#include "Tree.h"
#include "Node.h"

Tree::Tree()
{
	root = 0;
	nNode = nEffector = nJoint = 0;
}

void Tree::SetSeqNum(Node* node)
{
	switch (node->purpose) {
	case JOINT:
		node->seqNumJoint = nJoint++;
		node->seqNumEffector = -1;
		break;
	case EFFECTOR:
		node->seqNumJoint = -1;
		node->seqNumEffector = nEffector++;
		break;
	}
}

void Tree::InsertRoot(Node* root)
{
	assert( nNode==0 );
	nNode++;
	Tree::root = root;
	root->r = root->attach;
	assert( !(root->left || root->right) );
	SetSeqNum(root);
}

void Tree::InsertLeftChild(Node* parent, Node* child)
{
	assert(parent);
	nNode++;
	parent->left = child;
	child->realparent = parent;
	child->r = child->attach - child->realparent->attach;
	assert( !(child->left || child->right) );
	SetSeqNum(child);
}

void Tree::InsertRightSibling(Node* parent, Node* child)
{
	assert(parent);
	nNode++;
	parent->right = child;
	child->realparent = parent->realparent;
	child->r = child->attach - child->realparent->attach;
	assert( !(child->left || child->right) );
	SetSeqNum(child);
}

// Search recursively below "node" for the node with index value.
Node* Tree::SearchJoint(Node* node, int index)
{
	Node* ret;
	if (node != 0) {
		if (node->seqNumJoint == index) {
			return node;
		} else {
			if (ret = SearchJoint(node->left, index)) {
				return ret;
			}
			if (ret = SearchJoint(node->right, index)) {
				return ret;
			}
			return NULL;
		}
	} 
	else {
		return NULL;
	}
}


// Get the joint with the index value
Node* Tree::GetJoint(int index)
{
	return SearchJoint(root, index);
}

// Search recursively below node for the end effector with the index value
Node* Tree::SearchEffector(Node* node, int index)
{
	Node* ret;
	if (node != 0) {
		if (node->seqNumEffector == index) {
			return node;
		} else {
			if (ret = SearchEffector(node->left, index)) {
				return ret;
			}
			if (ret = SearchEffector(node->right, index)) {
				return ret;
			}
			return NULL;
		}
	} else {
		return NULL;
	}
}


// Get the end effector for the index value
Node* Tree::GetEffector(int index)
{
	return SearchEffector(root, index);
}

// Returns the global position of the effector.
const VectorR3& Tree::GetEffectorPosition(int index)
{
	Node* effector = GetEffector(index);
	assert(effector);
	return (effector->s);  
}

void Tree::ComputeTree(Node* node)
{
	if (node != 0) {
		node->ComputeS();
		node->ComputeW();
		ComputeTree(node->left);
		ComputeTree(node->right);
	}
}

void Tree::Compute(void)
{ 
	ComputeTree(root); 
}

void Tree::DrawTree(Node* node)
{
	if (node != 0) {
		glPushMatrix();
		node->DrawNode( node==root );	// Recursively draw node and update ModelView matrix
		if (node->left) {
			DrawTree(node->left);		// Draw tree of children recursively
		}
		glPopMatrix();
		if (node->right) {
			DrawTree(node->right);		// Draw right siblings recursively
		}
	}
}

void Tree::Draw(void) 
{
	DrawTree(root);
}

void Tree::PrintTree(Node* node)
{
	if (node != 0) {
		node->PrintNode();
		PrintTree(node->left);
		PrintTree(node->right);
	}
}

void Tree::Print(void) 
{ 
	PrintTree(root);  
	cout << "\n";
}

// Recursively initialize tree below the node
void Tree::InitTree(Node* node)
{
	if (node != 0) {
		node->InitNode();
		InitTree(node->left);
		InitTree(node->right);
	}
}

// Initialize all nodes in the tree
void Tree::Init(void)
{
	InitTree(root);
}

void Tree::UnFreezeTree(Node* node)
{
	if (node != 0) {
		node->UnFreeze();
		UnFreezeTree(node->left);
		UnFreezeTree(node->right);
	}
}

void Tree::UnFreeze(void)
{
	UnFreezeTree(root);
}