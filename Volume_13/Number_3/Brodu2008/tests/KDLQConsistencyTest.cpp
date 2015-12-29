#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
using namespace std;

#include <math.h>
#include <kdtree++/kdtree.hpp>

#include "lq.h"

typedef float FloatType;

struct Agent {
    int number;
    float coord[3];
    inline float& x() {return coord[0];}
    inline float& y() {return coord[1];}
    inline float& z() {return coord[2];}
    lqClientProxy* proxy;
    // KD tree interface
    typedef float value_type;
    inline float operator[](const int dim) const { return coord[dim]; }
};

typedef KDTree::KDTree<3, Agent> KD_Tree;

// Declare how many agents to put in the DB
const int Nagents = 10000;
const int Nqueries = 10000;
const int NmovesPerQuery = 5000;
Agent* agents;
vector<int> kdlqneighbors, sqneighbors;
FloatType sizeX, sizeY, sizeZ;

// There is no wrapping in KD and LQ
inline FloatType squaredDistance(FloatType dx,FloatType dy,FloatType dz){
    return dx*dx+dy*dy+dz*dz;
}

// The callback function for LQ - builds a vector of Agent* like NH
void callbackFunctionLQ(void* clientObject, float distanceSquared, void* clientQueryState) {
    reinterpret_cast<vector<int>*>(clientQueryState)->push_back(reinterpret_cast<Agent*>(clientObject)->number);
}

struct KDFunctor {
    vector<int>* nbagents;
    FloatType x, y, z, dsq;
    KDFunctor(vector<int>* nb, FloatType _x, FloatType _y, FloatType _z, FloatType _dsq) : nbagents(nb), x(_x), y(_y), z(_z), dsq(_dsq) {}
    // For the KD tree interface == back_inserter iterator
    KDFunctor& operator*() { return *this; }
    KDFunctor operator++(int) { return *this; }
    KDFunctor& operator=(const Agent& agent) {
        if (squaredDistance(agent[0] - x, agent[1] - y, agent[2] - z)<=dsq) nbagents->push_back(agent.number);
        return *this;
    }
};


// See neighand_structures.hpp. Allows compiler to track aliases unlike casts
union FloatConverter {
    FloatType f;
    uint32_t i;
    inline FloatConverter(FloatType _f) : f(_f) {}
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

    lqDB* db = lqCreateDatabase(10.f, 10.f, 10.f, 44.8, 44.8, 44.8, 32, 32, 32);

    agents = new Agent[Nagents];

    // +- 10 units on each side are outside
    for (int i=0; i<Nagents; ++i) {
        agents[i].number = i;
        agents[i].x() = 64.8 * (rand() / (RAND_MAX + 1.0f));
        agents[i].y() = 64.8 * (rand() / (RAND_MAX + 1.0f));
        agents[i].z() = 64.8 * (rand() / (RAND_MAX + 1.0f));

        // Insert in LQ
        agents[i].proxy = new lqClientProxy();
        lqInitClientProxy(agents[i].proxy, &agents[i]);
        lqUpdateForNewLocation(db, agents[i].proxy, agents[i].x(), agents[i].y(), agents[i].z());
    }

    bool globalFailed = false;
    int numULPdiff = 0;

    // Now call the callbackFunction on random queries with both, and compare
    // Use stl for sorting algorithms
    for (int query=0; query<Nqueries; ++query) {

        // First randomize the positions.
        // This tests that the move operation works as intended
        for (int j=0; j<NmovesPerQuery; ++j) {
            int a = rand() % Nagents;
            agents[a].x() = 64.8 * (rand() / (RAND_MAX + 1.0f));
            agents[a].y() = 64.8 * (rand() / (RAND_MAX + 1.0f));
            agents[a].z() = 64.8 * (rand() / (RAND_MAX + 1.0f));
            lqUpdateForNewLocation(db, agents[a].proxy, agents[a].x(), agents[a].y(), agents[a].z());
        }

        // Now perform a query
        // choose a distance that covers all case (completely out, intersect, englobes all)
        FloatType distance = 64.8f * (rand() / (RAND_MAX + 1.0f));
        FloatType distSquared = distance * distance;
        // choose point anywhere, possibly out
/*        FloatType x = 64.8f * (rand() / (RAND_MAX + 1.0f));
        FloatType y = 64.8f * (rand() / (RAND_MAX + 1.0f));
        FloatType z = 64.8f * (rand() / (RAND_MAX + 1.0f));
*/
        // Choose a center agent, possibly out
        int queryAgent = rand() % Nagents;
        FloatType x = agents[queryAgent].x();
        FloatType y = agents[queryAgent].y();
        FloatType z = agents[queryAgent].z();
        // Pass a build list as argument
        kdlqneighbors.clear(); sqneighbors.clear();

        const char* methodNames[2] = {"LQ", "KD"};
        int methodIdx = rand()%2;

        switch (methodIdx) {
            // LQ
            case 0:
                lqMapOverAllObjectsInLocality(db, x, y, z, distance, &callbackFunctionLQ, &kdlqneighbors);
                // simple brute algo
                for (int a = 0; a<Nagents; ++a) {
                    if (squaredDistance(agents[a].x()-x, agents[a].y()-y, agents[a].z()-z)<=distSquared) {
                        sqneighbors.push_back(a);
                    }
                }
                break;
            // KD
            case 1: {
                KD_Tree kdtree;
                for (int a = 0; a<Nagents; ++a) kdtree.insert( agents[a] );
                kdtree.optimize();
                //Agent virtualAgent; virtualAgent.x() = x; virtualAgent.y() = y; virtualAgent.z() = z;
                kdtree.find_within_range(agents[queryAgent], distance, KDFunctor(&kdlqneighbors, x, y, z, distance*distance));
                //for (unsigned int j=0; j<v.size(); ++j) kdlqneighbors.push_back(v[j].number);
                // simple brute algo
                for (int a = 0; a<Nagents; ++a) {
                    if (squaredDistance(agents[a].x()-x, agents[a].y()-y, agents[a].z()-z)<=distSquared) {
                        sqneighbors.push_back(a);
                    }
                }
                break;
            }
        }

        // post-process: sort the integers for comparison
        sort(kdlqneighbors.begin(), kdlqneighbors.end());
        sort(sqneighbors.begin(), sqneighbors.end());

        bool failed = false;

        // Automated test
        unsigned int kdlqidx = 0, sqidx = 0;
        stringstream sout;
        while ((kdlqidx < kdlqneighbors.size()) && (sqidx < sqneighbors.size())) {
            if (kdlqneighbors[kdlqidx] == sqneighbors[sqidx]) {++kdlqidx; ++sqidx; continue;}
            if (kdlqneighbors[kdlqidx] < sqneighbors[sqidx]) {
                FloatType dsq = squaredDistance(x-agents[kdlqneighbors[kdlqidx]].x(), y-agents[kdlqneighbors[kdlqidx]].y(), z-agents[kdlqneighbors[kdlqidx]].z());
                int32_t ulp = getULPDiff(dsq, distance*distance);
                if (abs(ulp)>ULP_precision) failed = true; else ++numULPdiff;
                sout << ulp << "\t" << kdlqneighbors[kdlqidx] << "\t" << "kdlq: dsq=" << dsq << " at (" << agents[kdlqneighbors[kdlqidx]].x() << ", " << agents[kdlqneighbors[kdlqidx]].y()<<", " <<agents[kdlqneighbors[kdlqidx]].z() << ")" <<endl;
                ++kdlqidx; continue;
            }
            FloatType dsq = squaredDistance(x-agents[sqneighbors[sqidx]].x(), y-agents[sqneighbors[sqidx]].y(), z-agents[sqneighbors[sqidx]].z());
            int32_t ulp = getULPDiff(dsq, distance*distance);
            if (abs(ulp)>ULP_precision) failed = true; else ++numULPdiff;
            sout << ulp << "\t" << sqneighbors[sqidx] << "\t" << "sq: dsq=" << dsq << " at (" << agents[sqneighbors[sqidx]].x() << ", " << agents[sqneighbors[sqidx]].y()<<", " <<agents[sqneighbors[sqidx]].z() << ")" <<endl;
            ++sqidx;
        }
        while (kdlqidx < kdlqneighbors.size()) {
            FloatType dsq = squaredDistance(x-agents[kdlqneighbors[kdlqidx]].x(), y-agents[kdlqneighbors[kdlqidx]].y(), z-agents[kdlqneighbors[kdlqidx]].z());
            int32_t ulp = getULPDiff(dsq, distance*distance);
            if (abs(ulp)>ULP_precision) failed = true; else ++numULPdiff;
            sout << ulp << "\t" << kdlqneighbors[kdlqidx] << "\t" << "kdlq: dsq=" << dsq << " at (" << agents[kdlqneighbors[kdlqidx]].x() << ", " << agents[kdlqneighbors[kdlqidx]].y()<<", " <<agents[kdlqneighbors[kdlqidx]].z() << ")" <<endl;
            ++kdlqidx;
        }
        while (sqidx < sqneighbors.size()) {
            FloatType dsq = squaredDistance(x-agents[sqneighbors[sqidx]].x(), y-agents[sqneighbors[sqidx]].y(), z-agents[sqneighbors[sqidx]].z());
            int32_t ulp = getULPDiff(dsq, distance*distance);
            if (abs(ulp)>ULP_precision) failed = true; else ++numULPdiff;
            sout << ulp << "\t" << sqneighbors[sqidx] << "\t" << "sq: dsq=" << dsq << " at (" << agents[sqneighbors[sqidx]].x() << ", " << agents[sqneighbors[sqidx]].y()<<", " <<agents[sqneighbors[sqidx]].z() << ")" <<endl;
            ++sqidx;
        }

        if (!sout.str().empty()) {
            cout << "FAILED for " << methodNames[methodIdx] << ", dsq="<<(distance*distance)<<", center=("<<x<<", "<<y<<", "<<z<<")"<<endl;
            cout << "ULP\tAgent#\tIn (sq/kdlq) list but not in the other list" << endl;
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


