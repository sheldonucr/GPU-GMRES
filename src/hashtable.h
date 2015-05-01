#ifndef __HASHTABLE_H
#define __HASHTABLE_H

#include "namepool.h"

class HashTable
{
  struct Member
  {
    int name;
    int addr;
    Member* next;
  };
  vector<Member> data;
  Member** table;
  namepool* pool;
  long int curIndex;

  long int HASH_TABLE_DIGIT;
  long int HASH_TABLE_SIZE;
  long int HASH_TABLE_MASK;

 public:
  HashTable(namepool* name, int digit=61); //constructor, table size = pow(2, digit)

  ~HashTable();

  unsigned int hashFunc(const char* name);   //hash function, return the hash key

  int find(const char* name);  //find if name is in the hashtable

  void insertAtCur(int name, int addr); //insert at the current index, which is set after find()

};

#endif
