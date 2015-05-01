/*
*******************************************************

        Cadence Extended Truncated Balanced Realization
                (*** CadETBR ***)

*******************************************************
*/

/*
 *    $RCSfile: namepool.cpp,v $
 *    $Revision: 1.1 $
 *    $Date: 2013/03/23 21:08:00 $
 *    Authors: Ning Mi
 *
 *    Functions: storage of name
 *
 */

#include <cstdio>
#include "namepool.h"


namepool::namepool(){
}

namepool::~namepool(){
}

int namepool::pushName(const char* name)
{
	int i;
	int addr;

	i=0;
	addr = pool.size();
	while(name[i]!='\0'){
		pool.push_back(name[i]);
		i++;
	}
	pool.push_back('\0');
	
	return addr;
}

bool namepool::getName(char* name, int addr)
{
	int i;

	if(addr>=pool.size()) return false;
	
	i = 0;

        while(pool[addr] != '\0'){
		name[i] = pool[addr];
		i++;
		addr++;
	}
	name[i] = '\0';
	
	return true;
}

bool namepool::isEqual(const char* name, int addr)
{
	int i;

	if(addr >= pool.size()) return false;

	i = 0;

	while(pool[addr] != '\0'){
		if (name[i++] != pool[addr++])
			return false;
	}
	if(name[i] != '\0') return false;

	return true;
}

bool namepool::printName(int addr, FILE* fid)
{	
	if(addr >= pool.size()) return false;

	if(fid == NULL){
		while(pool[addr]!='\0'){
			printf("%c",pool[addr]);
			addr++;
		}
		printf("\n");
	}
	else {
		while(pool[addr]!='\0'){
			fprintf(fid,"%c",pool[addr]);
			addr++;
		}
		fprintf(fid,"\n");
	}
	return true;
}
