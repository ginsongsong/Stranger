#include "float2str.h"

using namespace std;
string float2str( float &i )
{
   string s;
   stringstream ss( s );
   ss << i;
   return ss.str();
}