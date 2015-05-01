/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: namepool.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:00 $
 *    Authors: Ning Mi 
 * 
 *    Functions: storage of name 
 *
 */


#ifndef __NAMEPOOL_H
#define __NAMEPOOL_H

#include <cstdio>
#include <vector>
 
using namespace std;

class namepool
{
private:
	vector<char> pool;

public:
	namepool();
	~namepool();
	int pushName(const char* name);              //push a string into pool, return starting index
	bool getName(char* name, int addr);    //get the name from the addr
	bool isEqual(const char* name, int addr);    //compare name with pool[addr]
	bool printName(int addr, FILE* fid=NULL);    //print name to screen or file)
};

#endif
