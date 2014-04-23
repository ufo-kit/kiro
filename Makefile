CC=gcc
CFLAGS=-std=c99 -Wall -g -gdwarf-2 $(shell pkg-config --cflags gobject-2.0)
LDFLAGS= -lrdmacm -libverbs -lpthread $(shell pkg-config --libs gobject-2.0)


all: base

base: kiro-trb.o

kiro-cbr.o: kiro-trb.c kiro-trb.h
	$(CC) $(CFLAGS) $(LDFLAGS) -c kiro-trb.c -o kiro-trb.o


test-trb: test

test: kiro-trb.o test.c
	$(CC) $(CFLAGS) $(LDFLAGS) test.c kiro-trb.o -o test-tbr


clean:
	rm -f *.o test-trb



rebuild: clean all
	
