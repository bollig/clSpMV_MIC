TARGET = bandwidth-vectsum bandwidth-vectsum-pref bandwidth-simplesum-int bandwidth-simplesum-char bandwidth-memcpy bandwidth-memcpy-vect bandwidth-memset bandwidth-memset-nr bandwidth-memset-nrngo bandwidth-simplesum-char-O1 bandwidth-simplesum-int-O1

all: $(TARGET)

clean:
	rm $(TARGET)

include makefile.in


bandwidth-vectsum: bandwidth.cpp
	$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ bandwidth.cpp -D VECTSUMBASED $(LIBS) 

bandwidth-vectsum-pref: bandwidth.cpp
	$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ bandwidth.cpp -D VECTSUMBASED -D USEPREFETCH $(LIBS) 


bandwidth-simplesum-int-O1: bandwidth.cpp
	$(LD) $(CXXFLAGS) -O1 $(LDFLAGS) -o $@ bandwidth.cpp -D SIMPLESUMBASED -D INTBASED $(LIBS) 

bandwidth-simplesum-char-O1: bandwidth.cpp
	$(LD) $(CXXFLAGS) -O1 $(LDFLAGS) -o $@ bandwidth.cpp -D SIMPLESUMBASED -D CHARBASED $(LIBS) 

bandwidth-simplesum-int: bandwidth.cpp
	$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ bandwidth.cpp -D SIMPLESUMBASED -D INTBASED $(LIBS) 

bandwidth-simplesum-char: bandwidth.cpp
	$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ bandwidth.cpp -D SIMPLESUMBASED -D CHARBASED $(LIBS) 

bandwidth-memcpy: bandwidth.cpp
	$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ bandwidth.cpp -D MEMCOPYBASED $(LIBS) 

bandwidth-memcpy-vect: bandwidth.cpp
	$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ bandwidth.cpp -D MEMCOPYVECTBASED $(LIBS) 

bandwidth-memset: bandwidth.cpp
	$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ bandwidth.cpp -D MEMSETBASED $(LIBS) 

bandwidth-memset-nr: bandwidth.cpp
	$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ bandwidth.cpp -D MEMSETNOREAD $(LIBS) 

bandwidth-memset-nrngo: bandwidth.cpp
	$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ bandwidth.cpp -D MEMSETNOREADNOORDER $(LIBS) 


# DO NOT DELETE
