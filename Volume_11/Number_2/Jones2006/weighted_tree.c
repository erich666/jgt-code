/* weighted_tree.c - */

// Example code for "Efficient Generatino of Poisson-Disk Sampling
// Patterns," Thouis R. Jones, JGT vol. 11, No. 2, pp. 27-36
// 
// Copyright 2004-2006, Thouis R. Jones
// This code is distributed under the terms of the LGPL.


#include <math.h>
#include <stdlib.h>
#include "gts.h"

#define printf
#define g_assert(x)

#include "weighted_tree.h"

typedef struct {
  GtsVertex *v;
  double total_weight;
  double v_weight;
} treenode;

#define TREENODE(n) ((treenode *) ((n)->data))

gpointer new_tree(GtsVertex *v, double area)
{
  weighted_tree *wt = g_malloc(sizeof(weighted_tree));
  treenode *n = g_malloc(sizeof(treenode));

  n->v = v;
  n->v_weight = area;
  n->total_weight = area;

  wt->tree = g_node_new(n);
  wt->table = g_hash_table_new(NULL, NULL);
  g_hash_table_insert(wt->table, v, wt->tree);

  return (wt);
}

GtsVertex *select_vertex_from_tree_w(GNode *n, double w)
{
  treenode *tn = TREENODE(n);
  GNode *left_child = g_node_first_child(n);

  printf("selecting from tree with total weight %f\n", tn->total_weight);

  if ((w <= tn->v_weight) || (left_child == NULL))
    return (tn->v);

  w -= tn->v_weight;

  if (w <= TREENODE(left_child)->total_weight)
    return (select_vertex_from_tree_w(left_child,
                                      TREENODE(left_child)->total_weight * drand48()));

  if (left_child->next)
    return (select_vertex_from_tree_w(left_child->next,
                                      TREENODE(left_child->next)->total_weight * drand48()));

    return (select_vertex_from_tree_w(g_node_first_child(n), 
                                      TREENODE(left_child)->total_weight * drand48()));
}

GtsVertex *select_vertex_from_tree(weighted_tree *WT)
{
  return select_vertex_from_tree_w(WT->tree, TREENODE(WT->tree)->total_weight * drand48());
}

double weight_of_node(GNode *n)
{
  double w = TREENODE(n)->v_weight;
  if (g_node_first_child(n) != NULL) {
    w += TREENODE(g_node_first_child(n))->total_weight;
    if (g_node_first_child(n)->next != NULL) 
      w += TREENODE(g_node_first_child(n)->next)->total_weight;
  }

  return (w);
}


void add_to_tree_t(GNode *t, GNode *new_node)
{
  if (g_node_first_child(t) == NULL) {
    // make this the left (first) child of this node
    g_node_prepend(t, new_node);
  } else if (g_node_first_child(t)->next == NULL) {
    // make this the right (last) child of this node
    g_node_append(t, new_node);
  } else {
    if (drand48() > 0.5) {
      add_to_tree_t(g_node_first_child(t), new_node);
    } else {
      add_to_tree_t(g_node_first_child(t)->next, new_node);
    }
  }

  TREENODE(t)->total_weight = weight_of_node(t);
}

void add_to_tree(weighted_tree *WT, GtsVertex *v, double area)
{
  treenode *tn = g_malloc(sizeof(treenode));
  GNode *vn;

  tn->v = v;
  tn->v_weight = area;
  tn->total_weight = area;

  vn = g_node_new(tn);
  add_to_tree_t(WT->tree, vn);
  
  g_hash_table_insert(WT->table, v, vn);
}

void update_tree(weighted_tree *WT, GtsVertex *v, double new_area)
{
  GNode *n = g_hash_table_lookup(WT->table, v);
  
  // boundary vertices
  if (n == NULL) return;

  TREENODE(n)->v_weight = new_area;
  
  while (n) {
    TREENODE(n)->total_weight = weight_of_node(n);
    n = n->parent;
  }
}

#define IND for (i = 0; i < indent; i++) printf(" ");

void tree_area(GNode *n, int indent)
{
  int i;
  IND; printf("total  area: %lf\n", TREENODE(n)->total_weight);
  IND; printf("vertex area: %lf\n", TREENODE(n)->v_weight);
  if (g_node_first_child(n)) {
    IND; printf("left:");
    tree_area(g_node_first_child(n), indent + 2);
    if (g_node_first_child(n)->next) {
      IND; printf("right:");
      tree_area(g_node_first_child(n)->next, indent + 2);
    }
  }
}

int in_tree(weighted_tree *WT, GtsVertex *v)
{
  return (g_hash_table_lookup(WT->table, v) != NULL);
}

void print_vertex(gpointer key, gpointer data, gpointer user_data)
{
  GtsPoint *p = GTS_POINT(key);

  fprintf(stdout, "%lf %lf\n", p->x, p->y);
}

void print_verts(weighted_tree *WT)
{
  g_hash_table_foreach(WT->table, print_vertex, NULL);
}

static double exc;

void check_verts3(gpointer key, gpointer data, gpointer user_data)
{
  GtsPoint *p1 = GTS_POINT(key);
  GtsPoint *p2 = GTS_POINT(user_data);

  if (p1 == p2) return;

    printf("comparing %f %f and %f %f  %f\n",
           p1->x, p1->y, p2->x, p2->y, gts_point_distance(p1, p2));

  g_assert(gts_point_distance(p1, p2) >= exc);
}


void check_verts2(gpointer key, gpointer data, gpointer user_data)
{
  GtsPoint *p = GTS_POINT(key);
  weighted_tree *WT = (weighted_tree *) user_data;

  g_hash_table_foreach(WT->table, check_verts3, p);
}


void check_verts(weighted_tree *WT, double exclude_radius)
{
  exc = exclude_radius;

  g_hash_table_foreach(WT->table, check_verts2, WT);
}



int no_area_left(weighted_tree *WT)
{
  return (TREENODE(WT->tree)->total_weight <= 0.0);
}

double area_left(weighted_tree *WT)
{
  return (TREENODE(WT->tree)->total_weight);
}
