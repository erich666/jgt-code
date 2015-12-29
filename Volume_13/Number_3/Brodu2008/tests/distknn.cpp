/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This program compares how the Sphere and Cube performance
    for nearest neighbors queries evolves with distance.

    Nicolas Brodu, 2006/7
    Code released according to the GNU GPL, v2 or above.
*/

#include <sys/time.h>
#include <time.h>
#include <math.h>

#include <iostream>
#include <fstream>
using namespace std;

#include "neighand.h"
using namespace neighand;

// set these primary values

#ifndef BASE_EXP2
#warning Please define BASE_EXP2, WRAP_ARGUMENTS, and LOAD_RATIO
#define BASE_EXP2 4
//#define WRAP_ARGUMENTS true,true,true
#define WRAP_ARGUMENTS false,false,false
#endif

const FloatType loadRatio = 1.0f;
const int nmoves_base = 30;

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

int main() {

    cout << "Nearest neighbor test.\nInitializing (cell load = " << loadRatio << ", number of agents = " << Nagents << ")"<<endl;

    typedef NeighborhoodHandler<Agent,TEMPLATE_SIZE,WRAP_ARGUMENTS> NH;
    NH nh(10.f, 10.f, 10.f, cellSize);

    Agent* agents = new Agent[Nagents];

    // give the agents a unique number and an initial position
    for (int i=0; i<Nagents; ++i) {
        agents[i].number = i+1;
        agents[i].x() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
        agents[i].y() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
        agents[i].z() = 10.0f + worldSize*cellSize * (rand() / (RAND_MAX + 1.0f));
        // Insertion
        agents[i].proxy = nh.insert(agents[i].x(), agents[i].y(), agents[i].z(), &agents[i], ProxiedObjectRemapper<Agent>());
    }


    ofstream distknn("distknn.txt");

    // distance loop: process distances by 0.1 increment
    for (int di = 1; di <= 70; ++di) {
        float query_distance = di * 0.1f;

        cout << "Processing distance (in cell units): " << query_distance << endl;
        distknn << query_distance << flush;

        // scale to world units
        query_distance *= cellSize;

        // makes more moves for smaller dist to average over more multitasking time
        int nmoves = nmoves_base * 7 / di;

        timeval start, stop; float reprate;

        // main loop: simulate a movement of all agents, followed by agents querying their nearest neighbor.
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

            // each agent queries its nearest neighbor, avoiding the agent itself
            for (int i=0; i<Nagents; ++i) {
                NearestNeighbor<Agent> neighbor;
                nh.findNearestNeighbor(agents[i].proxy, query_distance, &neighbor);
            }

        }
        gettimeofday(&stop,0);
        reprate = getRate(start, stop, nmoves*Nagents);
        cout << "Sphere: " << reprate << flush;
        distknn << " " << reprate << flush;

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

            // each agent queries its nearest neighbor, avoiding the agent itself
            for (int i=0; i<Nagents; ++i) {
                NearestNeighbor<Agent> neighbor;
                nh.findNearestNeighbor(agents[i].proxy, query_distance, &neighbor);
            }

        }
        gettimeofday(&stop,0);
        reprate = getRate(start, stop, nmoves*Nagents);
        cout << ", Cube: " << reprate << flush;
        distknn << " " << reprate << flush;

        cout << endl;
        distknn << endl;
    }

    distknn.close();

    return 0;
}
