/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This program compares the different query methods performances

    Nicolas Brodu, 2006/7
    Code released according to the GNU GPL, v2 or above.
*/

#include <sys/time.h>
#include <time.h>
#include <math.h>

#include <iostream>
#include <fstream>
using namespace std;

#define NEIGHAND_SELECT_METHOD_STAT
#include "neighand.h"
using namespace neighand;

#include "lq.h"
#include <kdtree++/kdtree.hpp>

// set these primary values

#ifndef BASE_EXP2
#warning Please define BASE_EXP2, WRAP_ARGUMENTS, and LOAD_RATIO
#define BASE_EXP2 4
#define WRAP_ARGUMENTS true,true,true
//#define WRAP_ARGUMENTS false,false,false
#endif

const FloatType loadRatio = 5.0f;
const int nmoves = 30;

// derived values

const int worldSizeInt = 1 << BASE_EXP2;
const FloatType worldSize = worldSizeInt;
const int Nagents = int(loadRatio * worldSize * worldSize * worldSize);
const FloatType cellSize = 1.234f * worldSize / 16.f;
#define TEMPLATE_SIZE BASE_EXP2,BASE_EXP2,BASE_EXP2


struct Agent {
    int number;
    float coord[3];
    inline float& x() {return coord[0];}
    inline float& y() {return coord[1];}
    inline float& z() {return coord[2];}
    ObjectProxy<Agent>* proxy;
    // KD tree interface
    typedef float value_type;
    inline float operator[](const int dim) const { return coord[dim]; }
};

typedef KDTree::KDTree<3, Agent> KD_Tree;

// The callback function for LQ - builds a vector of Agent* like NH
void callbackFunctionLQ(void* clientObject, float distanceSquared, void* clientQueryState) {
    reinterpret_cast<vector<Agent*>* >(clientQueryState)->push_back(reinterpret_cast<Agent*>(clientObject));
}

static inline FloatType squaredDistance(FloatType dx,FloatType dy,FloatType dz){
    return dx*dx+dy*dy+dz*dz;
}

struct KDFunctor {
    vector<const Agent*>* vagents;
    FloatType x, y, z, dsq;
    KDFunctor(vector<const Agent*>* v, FloatType _x, FloatType _y, FloatType _z, FloatType _dsq) : vagents(v), x(_x), y(_y), z(_z), dsq(_dsq) {}
    // For the KD tree interface == back_inserter iterator
    KDFunctor& operator*() { return *this; }
    KDFunctor operator++(int) { return *this; }
    KDFunctor& operator=(const Agent& agent) {
        if (squaredDistance(agent[0] - x, agent[1] - y, agent[2] - z)<=dsq) vagents->push_back(&agent);
        return *this;
    }
};


FloatType getRate(timeval& start, timeval& stop, int reps = 1)
{
    long sec = stop.tv_sec - start.tv_sec;
    long usec = stop.tv_usec - start.tv_usec;
    if (usec<0) {
        sec -= 1;
        usec += 1000000L;
    }
    FloatType time = FloatType(sec) + FloatType(usec) * 1e-6f;
    return FloatType(reps) / time;
}

template<bool wrapx, bool wrapy, bool wrapz> struct MethodSwitcher {
    enum { SupportKDLQ = 0 };
};

// specialize for non-wrapping case only
template<> struct MethodSwitcher<false,false,false> {
    enum { SupportKDLQ = 1 };
};


int main() {

    cout << "Initializing (cell load = " << loadRatio << ", number of agents = " << Nagents << ")"<<endl;
    if (!MethodSwitcher<WRAP_ARGUMENTS>::SupportKDLQ) cout << "Note: The KD-Tree and LQ bin-lattice methods are always non-wrapping and not available in the wrapping cases" << endl;

    typedef NeighborhoodHandler<Agent,TEMPLATE_SIZE,WRAP_ARGUMENTS> NH;
    NH nh(10.f, 10.f, 10.f, cellSize);

    lqDB* db = lqCreateDatabase(10.f, 10.f, 10.f, worldSize*cellSize, worldSize*cellSize, worldSize*cellSize, worldSizeInt, worldSizeInt, worldSizeInt);

    Agent* agents = new Agent[Nagents];
    lqClientProxy* proxiesLQ = new lqClientProxy[Nagents];

    // give the agents a unique number and an initial position
    for (int i=0; i<Nagents; ++i) {
        agents[i].number = i+1;
        agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
        agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
        agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
    }

    // Insertion
    for (int i=0; i<Nagents; ++i) {
        agents[i].proxy = nh.insert(agents[i].x(), agents[i].y(), agents[i].z(), &agents[i], ProxiedObjectRemapper<Agent>());
        lqInitClientProxy(&proxiesLQ[i], &agents[i]);
        lqUpdateForNewLocation(db, &proxiesLQ[i], agents[i].x(), agents[i].y(), agents[i].z());
    }


    float dparm[3];
    dparm[0] = 0.8f;                         // close range below one cell
    dparm[1] = (worldSizeInt/2 -1) * 0.5f;   // medium range
    dparm[2] = (worldSizeInt/2 -1);          // large range

    ofstream histogram("performances.txt");

    // distance loop: process three distances
    for (int didx = 0; didx < 3; ++didx) {
        float query_distance = dparm[didx];

        cout << "Processing distance (in cell units): " << query_distance << endl;
        histogram << query_distance << flush;

        // scale to world units
        query_distance *= cellSize;

        timeval start, stop; float reprate;

        // Compute average factors for combi
        srand(42); nh.setQueryMethod(Auto);
        float avgCube=0.0f, avgSphere=0.0f, avgBrute=0.0f, avgNonEmpty=0.0f;
        for (int i=0; i<Nagents; ++i) {
            agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
            agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
            agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
        }
        float factorCube, factorSphere, factorBrute, factorNonEmpty;
        for (int i=0; i<Nagents; ++i) {
            nh.getAutoFactors(agents[i].x(), agents[i].y(), agents[i].z(), query_distance, factorSphere, factorCube, factorNonEmpty, factorBrute);
            avgCube += factorCube; avgSphere += factorSphere; avgBrute += factorBrute; avgNonEmpty += factorNonEmpty;
        }
        avgCube /= Nagents; avgSphere /= Nagents; avgBrute /= Nagents; avgNonEmpty /= Nagents;
        cout << "Auto-estimated average cost factors: S=" << avgSphere << ", C=" << avgCube << ", B=" << avgBrute << ", N=" << avgNonEmpty << endl;

        nh.resetStat();

        // main loop: simulate a movement of all agents, followed by agents querying their neighbors.
        // do it for each method with the same seed
        nh.setQueryMethod(Auto);
        srand(42); gettimeofday(&start,0);
        for (int movestep = 0; movestep<nmoves; ++movestep) {

            // update the agents positions
            for (int i=0; i<Nagents; ++i) {
                agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
            }

            // Update the structures
            for (int i=0; i<Nagents; ++i) nh.update(agents[i].proxy, agents[i].x(), agents[i].y(), agents[i].z());

            // each agent queries its neighbors
            for (int i=0; i<Nagents; ++i) {
                vector<Agent*> neighbors;
                nh.findNeighbors(agents[i].x(), agents[i].y(), agents[i].z(), query_distance, neighbors);
            }

        }
        gettimeofday(&stop,0);
        reprate = getRate(start, stop, nmoves*Nagents);
        cout << "Auto: " << reprate << flush;
        histogram << " " << reprate << flush;

        cout << " (calls:S=" << nh.statSphere << ",C=" << nh.statCube << ",B=" << nh.statBrute << ",N=" << nh.statNonEmpty << ")" << flush;

        nh.setQueryMethod(Sphere);
        srand(42); gettimeofday(&start,0);
        for (int movestep = 0; movestep<nmoves; ++movestep) {

            // update the agents positions
            for (int i=0; i<Nagents; ++i) {
                agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
            }

            // Update the structures
            for (int i=0; i<Nagents; ++i) nh.update(agents[i].proxy, agents[i].x(), agents[i].y(), agents[i].z());

            // each agent queries its neighbors
            for (int i=0; i<Nagents; ++i) {
                vector<Agent*> neighbors;
                nh.findNeighbors(agents[i].x(), agents[i].y(), agents[i].z(), query_distance, neighbors);
            }

        }
        gettimeofday(&stop,0);
        reprate = getRate(start, stop, nmoves*Nagents);
        cout << ", Sphere: " << reprate << flush;
        histogram << " " << reprate << flush;

        nh.setQueryMethod(Cube);
        srand(42); gettimeofday(&start,0);
        for (int movestep = 0; movestep<nmoves; ++movestep) {

            // update the agents positions
            for (int i=0; i<Nagents; ++i) {
                agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
            }

            // Update the structures
            for (int i=0; i<Nagents; ++i) nh.update(agents[i].proxy, agents[i].x(), agents[i].y(), agents[i].z());

            // each agent queries its neighbors
            for (int i=0; i<Nagents; ++i) {
                vector<Agent*> neighbors;
                nh.findNeighbors(agents[i].x(), agents[i].y(), agents[i].z(), query_distance, neighbors);
            }

        }
        gettimeofday(&stop,0);
        reprate = getRate(start, stop, nmoves*Nagents);
        cout << ", Cube: " << reprate << flush;
        histogram << " " << reprate << flush;

        nh.setQueryMethod(NonEmpty);
        srand(42); gettimeofday(&start,0);
        for (int movestep = 0; movestep<nmoves; ++movestep) {

            // update the agents positions
            for (int i=0; i<Nagents; ++i) {
                agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
            }

            // Update the structures
            for (int i=0; i<Nagents; ++i) nh.update(agents[i].proxy, agents[i].x(), agents[i].y(), agents[i].z());

            // each agent queries its neighbors
            for (int i=0; i<Nagents; ++i) {
                vector<Agent*> neighbors;
                nh.findNeighbors(agents[i].x(), agents[i].y(), agents[i].z(), query_distance, neighbors);
            }

        }
        gettimeofday(&stop,0);
        reprate = getRate(start, stop, nmoves*Nagents);
        cout << ", Non-Empty: " << reprate << flush;
        histogram << " " << reprate << flush;

        nh.setQueryMethod(Brute);
        srand(42); gettimeofday(&start,0);
        for (int movestep = 0; movestep<nmoves; ++movestep) {

            // update the agents positions
            for (int i=0; i<Nagents; ++i) {
                agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
            }

            // Update the structures
            for (int i=0; i<Nagents; ++i) nh.update(agents[i].proxy, agents[i].x(), agents[i].y(), agents[i].z());

            // each agent queries its neighbors
            for (int i=0; i<Nagents; ++i) {
                vector<Agent*> neighbors;
                nh.findNeighbors(agents[i].x(), agents[i].y(), agents[i].z(), query_distance, neighbors);
            }

        }
        gettimeofday(&stop,0);
        reprate = getRate(start, stop, nmoves*Nagents);
        cout << ", Brute: " << reprate << flush;
        histogram << " " << reprate << flush;

        srand(42); gettimeofday(&start,0);
        for (int movestep = 0; movestep<nmoves; ++movestep) {

            // update the agents positions
            for (int i=0; i<Nagents; ++i) {
                agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
            }

            // each agent queries its neighbors
            for (int i=0; i<Nagents; ++i) {
                for (int j=0; j<Nagents; ++j) {
                    vector<Agent*> neighbors;
                    // use of the NH function only for wrapping.
                    // The squared dist is simply dx*dx+dy*dy+dz*dz, inlined, when no wrapping
                    if (nh.squaredDistance(
                    (agents[i].x() - agents[j].x()),
                    (agents[i].y() - agents[j].y()),
                    (agents[i].z() - agents[j].z())
                    ) <= query_distance * query_distance) {
                        neighbors.push_back(&agents[j]);
                    }
                }
            }

        }
        gettimeofday(&stop,0);
        reprate = getRate(start, stop, nmoves*Nagents);
        cout << ", brute(raw): " << reprate << flush;
        histogram << " " << reprate << flush;

        if (MethodSwitcher<WRAP_ARGUMENTS>::SupportKDLQ) {
            srand(42); gettimeofday(&start,0);
            for (int movestep = 0; movestep<nmoves; ++movestep) {

                // update the agents positions
                for (int i=0; i<Nagents; ++i) {
                    agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                    agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                    agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                }

                // store in kd-tree: since all points have moved, individual erase()/insert() are inefficient and the whole tree needs to be rebuilt
                KD_Tree kdtree;
                for (int i=0; i<Nagents; ++i) kdtree.insert( agents[i] );
                kdtree.optimize();

                // each agent queries its neighbors
                for (int i=0; i<Nagents; ++i) {
                    std::vector<const Agent*> v;
                    kdtree.find_within_range(agents[i], query_distance, KDFunctor(&v, agents[i].x(), agents[i].y(), agents[i].z(), query_distance*query_distance));
                }

            }
            gettimeofday(&stop,0);
            reprate = getRate(start, stop, nmoves*Nagents);
            cout << ", KD-tree: " << reprate << flush;
            histogram << " " << reprate << flush;

            srand(42); gettimeofday(&start,0);
            for (int movestep = 0; movestep<nmoves; ++movestep) {

                // update the agents positions
                for (int i=0; i<Nagents; ++i) {
                    agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                    agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                    agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
                }

                // Update the structures
                for (int i=0; i<Nagents; ++i) lqUpdateForNewLocation(db, &proxiesLQ[i], agents[i].x(), agents[i].y(), agents[i].z());

                // each agent queries its neighbors
                for (int i=0; i<Nagents; ++i) {
                    vector<const Agent*> neighbors;
                    lqMapOverAllObjectsInLocality(db, agents[i].x(), agents[i].y(), agents[i].z(), query_distance, &callbackFunctionLQ, &neighbors);
                }

            }
            gettimeofday(&stop,0);
            reprate = getRate(start, stop, nmoves*Nagents);
            cout << ", bin-lattice(LQ): " << reprate << flush;
            histogram << " " << reprate << flush;
        }

        cout << endl;
        histogram << endl;
    }

    histogram.close();

    return 0;
}
