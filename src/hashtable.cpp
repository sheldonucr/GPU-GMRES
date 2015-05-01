#include "hashtable.h"
#include <stdlib.h>

HashTable::HashTable(namepool* name, int digit)
{
  int i;
  HASH_TABLE_DIGIT = digit;
  HASH_TABLE_SIZE = 1 << HASH_TABLE_DIGIT;
  HASH_TABLE_MASK = HASH_TABLE_SIZE - 1;
  table = (Member**)malloc(HASH_TABLE_SIZE*sizeof(Member*));
  for (i = 0; i<HASH_TABLE_SIZE; i++){
    table[i] == NULL;
  }
  pool = name;
}

HashTable::~HashTable()
{
  free(table);
}

unsigned int HashTable::hashFunc(const char* name)
{
  int cnt = 1, mul;
  const char* ch = name;
  unsigned register hashno = 1;

  hashno = 1;
  ch++;

  if((*ch)=='\0') hashno=(*(ch-1));

  while((*ch)!='\0'){
    mul = (*ch)*(*(ch-1))*cnt;
    hashno += mul;
    cnt++;
    if(cnt > 20) cnt=1;
    ch++;
  }
  return hashno;
}

int HashTable::find(const char* name)
{
  curIndex = hashFunc(name) & HASH_TABLE_MASK;

  Member* tempMember = table[curIndex];

  while( tempMember != NULL )
    {
      if (pool->isEqual(name, tempMember->name)) return tempMember->addr;
      tempMember = tempMember->next;
    }
  return -1;
}

void HashTable::insertAtCur(int name, int addr)
{
  Member *tempMember = (Member*)malloc(sizeof(Member));
  tempMember->name = name;
  tempMember->addr = addr;
  tempMember->next = table[curIndex];
  table[curIndex] = tempMember;
}


