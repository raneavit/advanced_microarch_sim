#ifndef CIRCULARFIFO_H
#define CIRCULARFIFO_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <cassert>
using namespace std;

template <typename T> class CircularFIFO {
private:
    
 
public:
    T* fifoList;
    uint32_t listSize;
    int head, tail ;
    bool headPhase, tailPhase;

    CircularFIFO(uint32_t size);
    uint32_t getFreeEntries();
    uint32_t getNoOfEntries();
    void push(T val);
    T pop();
};

template <typename T> CircularFIFO<T>::CircularFIFO(uint32_t size)
{
    listSize = size;
    fifoList = (T*) malloc(listSize * sizeof(T));
    head = 0;
    tail = 0;
    headPhase = true;
    tailPhase = true;
}

template <typename T> uint32_t CircularFIFO<T>::getFreeEntries()
{
    uint32_t result;
    if(headPhase == tailPhase) result = listSize - tail + head;
    else result = head - tail;   
    return result;
}

template <typename T> uint32_t CircularFIFO<T>::getNoOfEntries()
{
    uint32_t result;
    if(headPhase == tailPhase) result = tail - head;
    else result = listSize - head + tail;   
    return result;
}

template <typename T> void CircularFIFO<T>::push(T val)
{
    fifoList[tail] = val;
    tail++;
    if(tail == listSize){
        tail = 0;
        tailPhase = !tailPhase;
    }  
}

template <typename T> T CircularFIFO<T>::pop()
{
    T result;
    result = fifoList[head];
    head++;
    if(head == listSize){
        head = 0;
        headPhase = !headPhase;
    }
    return result;   
}

#endif