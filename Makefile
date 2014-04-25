CC=gcc
CFLAGS=-std=c99 -Wall -g -gdwarf-2 $(shell pkg-config --cflags gobject-2.0)
LDFLAGS= -lrdmacm -libverbs -lpthread $(shell pkg-config --libs gobject-2.0)


.PHONY : all
all: base

base: kiro-trb.o kiro-client.o kiro-server.o

kiro-trb.o: kiro-trb.c kiro-trb.h
	$(CC) $(CFLAGS) $(LDFLAGS) -c kiro-trb.c -o kiro-trb.o

kiro-client.o: kiro-client.c kiro-client.h
	$(CC) $(CFLAGS) $(LDFLAGS) -c kiro-client.c -o kiro-client.o

kiro-server.o: kiro-server.c kiro-server.h
	$(CC) $(CFLAGS) $(LDFLAGS) -c kiro-server.c -o kiro-server.o


.PHONY : test
test: test-trb client server

test-trb: kiro-trb.o test.c
	$(CC) $(CFLAGS) $(LDFLAGS) test.c kiro-trb.o -o test-trb

client: kiro-client.o test-client.c
	$(CC) $(CFLAGS) $(LDFLAGS) test-client.c kiro-client.o -o client

server: kiro-server.o test-server.c
	$(CC) $(CFLAGS) $(LDFLAGS) test-server.c kiro-server.o -o server


.PHONY : clean
clean:
	rm -f *.o test-trb client server


.PHONY : rebuild
rebuild: clean all
	
