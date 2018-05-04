#include "int2str.h"

using namespace std;
string int2str( int &i )
{
    string s;
    stringstream ss( s );
    ss << i;
    return ss.str();
}