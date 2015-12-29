/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This program checks that the library works as intended

    Nicolas Brodu, 2006/7
    Code released according to the GNU GPL, v2 or above.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
using namespace std;

#include <math.h>

#include "neighand.h"
using namespace neighand;


struct Agent {
    int number;
    FloatType x,y,z; // to find back the position for SQ
    ObjectProxy<Agent>* proxy;
};

// Declare how many agents to put in the DB
const int Nagents = 10000;
Agent* agents;
vector<int> nhneighbors, sqneighbors;
FloatType sizeX, sizeY, sizeZ;

template <bool wrapX, bool wrapY, bool wrapZ>
inline FloatType squaredDistance(FloatType dx,FloatType dy,FloatType dz);

template <>
inline FloatType squaredDistance<false,false,false>(FloatType dx,FloatType dy,FloatType dz)
{
    return dx*dx+dy*dy+dz*dz;
}

template <>
inline FloatType squaredDistance<true,true,false>(FloatType dx,FloatType dy,FloatType dz)
{
    dx = remainderf(dx, sizeX);
    dy = remainderf(dy, sizeY);
    return dx*dx+dy*dy+dz*dz;
}

template <>
inline FloatType squaredDistance<true,true,true>(FloatType dx,FloatType dy,FloatType dz)
{
    dx = remainderf(dx, sizeX);
    dy = remainderf(dy, sizeY);
    dz = remainderf(dz, sizeZ);
    return dx*dx+dy*dy+dz*dz;
}

struct VectorFunctor {
    vector<int>* nbagents;
    VectorFunctor(vector<int>* nb) : nbagents(nb) {}
    void operator()(Agent* agent) {
        nbagents->push_back(agent->number);
    }
};


int32_t getULPDiff(FloatConverter x, FloatConverter y) {
    return x.i - y.i;
}

void help() {
cout <<
"Usage:\n"
"-s seed        Use the given random seed\n"
"-u ULP         Use the given Units of Least Precision tolerance for float comparisons\n"
"-h             This help\n"
<< flush;
}

int main(int argc, char** argv) {

    unsigned long seed = 0;
    int32_t ULP_precision = 0;

    int c; while ((c=getopt(argc,argv,"hs:u:"))!=-1) switch(c) {
        case 's': seed = atoi(optarg); break;
        case 'u': ULP_precision = atoi(optarg); break;
        default: help(); return 0;
    }

    if (seed==0) seed = time(0);

    cout << "Using random seed: " << seed << " (pass as argument to reproduce this test)." << endl;
    if (ULP_precision) cout << "Using " << ULP_precision << " ULP for float comparison tolerance." << endl;
    else cout << "Using exact match for float comparisons, see the -u option for setting a tolerance." << endl;
    srand(seed);

    // 32x32x32 bloc, from 10 to 54.8 (random values)
    typedef NeighborhoodHandler<Agent,5,5,5,CONSISTENCY_TEST_WRAP> NH;
    NH nh(10.f, 10.f, 10.f, 1.4f);//, "consistency.bin");
    sizeX = 44.8;
    sizeY = 44.8;
    sizeZ = 44.8;

    agents = new Agent[Nagents];

    // Put them in +- 10 units on each side are outside
    for (int i=0; i<Nagents; ++i) {
        agents[i].number = i;
        agents[i].x = 64.8 * (rand() / (RAND_MAX + 1.0f));
        agents[i].y = 64.8 * (rand() / (RAND_MAX + 1.0f));
        agents[i].z = 64.8 * (rand() / (RAND_MAX + 1.0f));

        // Insert in NH
        agents[i].proxy = nh.insert(agents[i].x, agents[i].y, agents[i].z, &agents[i], ProxiedObjectRemapper<Agent>());
    }

    int Nqueries = 10000;
    int NmovesPerQuery = 5000;
    bool globalFailed = false;
    int numULPdiff = 0;

    // Now call the callbackFunction on random queries with both, and compare
    // Use stl for sorting algorithms
    for (int i=0; i<Nqueries; ++i) {

        // First randomize the positions.
        // This tests that the move operation works as intended
        for (int j=0; j<NmovesPerQuery; ++j) {
            int a = rand() % Nagents;
            agents[a].x = 64.8 * (rand() / (RAND_MAX + 1.0f));
            agents[a].y = 64.8 * (rand() / (RAND_MAX + 1.0f));
            agents[a].z = 64.8 * (rand() / (RAND_MAX + 1.0f));
            nh.update(agents[a].proxy, agents[a].x, agents[a].y, agents[a].z);
        }

        // Now perform a query
        // choose a distance that covers all case (completely out, intersect, englobes all)
        FloatType distance = 64.8f * (rand() / (RAND_MAX + 1.0f));
        FloatType distSquared = distance * distance;
        // choose point anywhere, possibly out
        FloatType x = 64.8f * (rand() / (RAND_MAX + 1.0f));
        FloatType y = 64.8f * (rand() / (RAND_MAX + 1.0f));
        FloatType z = 64.8f * (rand() / (RAND_MAX + 1.0f));
        // Pass a build list as argument
        nhneighbors.clear(); sqneighbors.clear();

        NearestNeighbor<Agent> closest;
        NearestNeighbor<Agent> knn[20]; int knnfound;

        QueryMethod methods[4] = {Sphere, Cube, NonEmpty, Brute};
        const char* methodNames[4] = {"Sphere", "Cube", "NonEmpty", "Brute"};
        int methodIdx = rand()%4;
        QueryMethod method = methods[methodIdx];
        nh.setQueryMethod(method);

        int closestSelected = rand()%3;
        switch (closestSelected) {
            // find all neighbors
            case 0:
                nh.applyToNeighbors(x, y, z, distance, VectorFunctor(&nhneighbors));
                // simple query algo
                for (int a = 0; a<Nagents; ++a) {
                    if (squaredDistance<CONSISTENCY_TEST_WRAP>(agents[a].x-x, agents[a].y-y, agents[a].z-z)<=distSquared) {
                        sqneighbors.push_back(a);
                    }
                }
                break;
            // Nearest neighbor only.
            case 1:
                if (nh.findNearestNeighbor(x, y, z, distance, &closest)!=0) nhneighbors.push_back(closest.object->number);
                { FloatType dmin = distSquared; int amin = -1;
                // simple query algo
                for (int a = 0; a<Nagents; ++a) {
                    FloatType dsq=squaredDistance<CONSISTENCY_TEST_WRAP>(agents[a].x-x, agents[a].y-y, agents[a].z-z);
                    if (dsq <= dmin) {
                        dmin = dsq;
                        amin = a;
                    }
                }
                if (amin!=-1) sqneighbors.push_back(amin);
                }
                break;
            // find the K nearest neighbors, reasonable random K range
            case 2: {
                int N = 1+(rand()%20);
                knnfound = nh.findNearestNeighbors(x, y, z, distance, knn, N);
                for (int i=0; i<knnfound; ++i) nhneighbors.push_back(knn[i].object->number);
                // sq now
                {
                    for (int i=0; i<N; ++i) knn[i].squaredDistance = numeric_limits<float>::max();
                    int nfound = 0;
                    for (int a = 0; a<Nagents; ++a) {
                        FloatType dsq=squaredDistance<CONSISTENCY_TEST_WRAP>(agents[a].x-x, agents[a].y-y, agents[a].z-z);
                        if (dsq <= distSquared) {
                            if (++nfound>N) nfound=N;
                            NearestNeighbor<Agent> currentObject;
                            currentObject.squaredDistance = dsq;
                            currentObject.object = &agents[a];
                            // keep only the min. don't reduce search distance here
                            for (int i=0; i<nfound; ++i) if (dsq < knn[i].squaredDistance) {
                                NearestNeighbor<Agent> tmp = knn[i];
                                knn[i] = currentObject;
                                currentObject = tmp;
                            }
                        }
                    }
                    for (int i=0; i<nfound; ++i) sqneighbors.push_back(knn[i].object->number);
                }
                break;
            }
        }


//        showLists(x,y,z,distance);

        // post-process: sort the integers for comparison
        sort(nhneighbors.begin(), nhneighbors.end());
        sort(sqneighbors.begin(), sqneighbors.end());

        bool failed = false;

        // Automated test
        unsigned int nhidx = 0, sqidx = 0;
        stringstream sout;
        while ((nhidx < nhneighbors.size()) && (sqidx < sqneighbors.size())) {
            if (nhneighbors[nhidx] == sqneighbors[sqidx]) {++nhidx; ++sqidx; continue;}
            if (nhneighbors[nhidx] < sqneighbors[sqidx]) {
                FloatType dsq = squaredDistance<CONSISTENCY_TEST_WRAP>(x-agents[nhneighbors[nhidx]].x, y-agents[nhneighbors[nhidx]].y, z-agents[nhneighbors[nhidx]].z);
                int32_t ulp = getULPDiff(dsq, distance*distance);
                if (abs(ulp)>ULP_precision) failed = true; else ++numULPdiff;
                sout << ulp << "\t" << nhneighbors[nhidx] << "\t" << "nh: dsq=" << dsq << " at (" << agents[nhneighbors[nhidx]].x << ", " << agents[nhneighbors[nhidx]].y<<", " <<agents[nhneighbors[nhidx]].z << ")" <<endl;
                ++nhidx; continue;
            }
            FloatType dsq = squaredDistance<CONSISTENCY_TEST_WRAP>(x-agents[sqneighbors[sqidx]].x, y-agents[sqneighbors[sqidx]].y, z-agents[sqneighbors[sqidx]].z);
            int32_t ulp = getULPDiff(dsq, distance*distance);
            if (abs(ulp)>ULP_precision) failed = true; else ++numULPdiff;
            sout << ulp << "\t" << sqneighbors[sqidx] << "\t" << "sq: dsq=" << dsq << " at (" << agents[sqneighbors[sqidx]].x << ", " << agents[sqneighbors[sqidx]].y<<", " <<agents[sqneighbors[sqidx]].z << ")" <<endl;
            ++sqidx;
        }
        while (nhidx < nhneighbors.size()) {
            FloatType dsq = squaredDistance<CONSISTENCY_TEST_WRAP>(x-agents[nhneighbors[nhidx]].x, y-agents[nhneighbors[nhidx]].y, z-agents[nhneighbors[nhidx]].z);
            int32_t ulp = getULPDiff(dsq, distance*distance);
            if (abs(ulp)>ULP_precision) failed = true; else ++numULPdiff;
            sout << ulp << "\t" << nhneighbors[nhidx] << "\t" << "nh: dsq=" << dsq << " at (" << agents[nhneighbors[nhidx]].x << ", " << agents[nhneighbors[nhidx]].y<<", " <<agents[nhneighbors[nhidx]].z << ")" <<endl;
            ++nhidx;
        }
        while (sqidx < sqneighbors.size()) {
            FloatType dsq = squaredDistance<CONSISTENCY_TEST_WRAP>(x-agents[sqneighbors[sqidx]].x, y-agents[sqneighbors[sqidx]].y, z-agents[sqneighbors[sqidx]].z);
            int32_t ulp = getULPDiff(dsq, distance*distance);
            if (abs(ulp)>ULP_precision) failed = true; else ++numULPdiff;
            sout << ulp << "\t" << sqneighbors[sqidx] << "\t" << "sq: dsq=" << dsq << " at (" << agents[sqneighbors[sqidx]].x << ", " << agents[sqneighbors[sqidx]].y<<", " <<agents[sqneighbors[sqidx]].z << ")" <<endl;
            ++sqidx;
        }

        if (!sout.str().empty()) {
            cout << "FAILED for " << methodNames[methodIdx] << " " << (closestSelected?((closestSelected==1)?"(find closest)":"(find K nearest)"):"(find all)") << ", dsq="<<(distance*distance)<<", center=("<<x<<", "<<y<<", "<<z<<")"<<endl;
            cout << "ULP\tAgent#\tIn (sq/nh) list but not in the other list" << endl;
            cout << sout.str();
            globalFailed = true;
        }
    }

    if (!globalFailed) {
        cout << Nqueries << " random tests OK";
        if (numULPdiff>0) cout << " (" << numULPdiff << " discrepancies below or equal to "<< ULP_precision << " ULP)";
        else cout << " (perfect match)";
        cout << endl;
    }
    else {
        cout << endl << "Check FAILED. See the ULP numbers above to check if this is a real failure or a float point comparison discrepancy. If so, use the -u option to set the units of least precision accuracy which is acceptable to your application (default is exact floating-point match). If this is a real failure, then check your compilation options and if the problem persists please report a bug." << endl;
        return 1;
    }
    return 0;
}


