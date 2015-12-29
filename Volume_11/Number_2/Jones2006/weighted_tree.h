// weighted_tree.h - 




// Example code for "Efficient Generatino of Poisson-Disk Sampling
// Patterns," Thouis R. Jones, JGT vol. 11, No. 2, pp. 27-36
// 
// Copyright 2004-2006, Thouis R. Jones
// This code is distributed under the terms of the LGPL.

typedef struct {
  GNode *tree;
  GHashTable *table; // vetex to node hash table
} weighted_tree;

gpointer new_tree(GtsVertex *v, double area);
GtsVertex *select_vertex_from_tree(weighted_tree *WT);
void add_to_tree(weighted_tree *WT, GtsVertex *v, double area);
void update_tree(weighted_tree *WT, GtsVertex *v, double new_area);
int in_tree(weighted_tree *WT, GtsVertex *v);
void print_verts(weighted_tree *WT);
int no_area_left(weighted_tree *WT);
void check_verts(weighted_tree *WT, double exclude_radius);
double area_left(weighted_tree *WT);
