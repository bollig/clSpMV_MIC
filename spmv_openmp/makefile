
INCLUDES   = -I./

TARGET     =  seq-spmv omp-spmv omp-spmv-pref64 omp-spmv-pref128 omp-spmv-pref256 omp-spmv-pref512 omp-spmv-pref1024 omp-spmv-vect-row-88 omp-spmv-vect-row-48 omp-spmv-vect-row-28 omp-spmv-vect-row-18 omp-spmv-vect-row2-88 omp-spmv-vect-row2-48 omp-spmv-vect-row2-28 omp-spmv-vect-row2-18 omp-spmv-vect-col-88 omp-spmv-vect-col-84 omp-spmv-vect-col-82 omp-spmv-vect-col-81 omp-spmv-vect-swi-88 omp-spmv-vect  omp-spmv-pref omp-spmm mkl-spmv mkl-spmm mkl-tspmv omp-spmm-vect 
#TARGET     =  seq-spmv omp-spmv omp-spmv-pref64 omp-spmv-pref128 omp-spmv-pref256 omp-spmv-pref512 omp-spmv-pref1024 omp-spmv-vect-row-88 omp-spmv-vect-row-48 omp-spmv-vect-row-28 omp-spmv-vect-row-18 omp-spmv-vect-row2-88 omp-spmv-vect-row2-48 omp-spmv-vect-row2-28 omp-spmv-vect-row2-18 omp-spmv-vect-col-88 omp-spmv-vect-col-84 omp-spmv-vect-col-82 omp-spmv-vect-col-81 omp-spmv-vect-swi-88 omp-spmv-pref omp-spmm mkl-spmv mkl-spmm mkl-tspmv omp-spmm-vect 

#TARGET = omp-spmm omp-spmm-vect-reg omp-spmm-vect-nr omp-spmm-vect-nrngo omp-spmm-vect

CFILES       = ulib 

CXXFILES   = seq-spmv omp-spmv omp-spmv-vect omp-spmv-pref omp-spmm omp-spmm-vect 

SRCS	= $(CFILES:%=%.c) $(CXXFILES:%=%.cpp)



all: $(TARGET)
	
	

installCPU: $(TARGET)
	cp $(TARGET) ../../CPU/

installKNC: $(TARGET)
	cp $(TARGET) ../../KNC/


seq-spmv: seq-spmv.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o seq-spmv.o $(LIBS) 

omp-spmv: omp-spmv.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv.o $(LIBS) 

omp-spmv-pref64.o: omp-spmv.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPREFETCHDISTANCE=64 -c omp-spmv.cpp -o $@

omp-spmv-pref64: omp-spmv-pref64.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-pref64.o $(LIBS) 

omp-spmv-pref128.o: omp-spmv.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPREFETCHDISTANCE=128 -c omp-spmv.cpp -o $@

omp-spmv-pref128: omp-spmv-pref128.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-pref128.o $(LIBS) 

omp-spmv-pref256.o: omp-spmv.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPREFETCHDISTANCE=256 -c omp-spmv.cpp -o $@

omp-spmv-pref256: omp-spmv-pref256.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-pref256.o $(LIBS) 

omp-spmv-pref512.o: omp-spmv.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPREFETCHDISTANCE=512 -c omp-spmv.cpp -o $@

omp-spmv-pref512: omp-spmv-pref512.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-pref512.o $(LIBS) 

omp-spmv-pref1024.o: omp-spmv.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPREFETCHDISTANCE=1024 -c omp-spmv.cpp -o $@

omp-spmv-pref1024: omp-spmv-pref1024.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-pref1024.o $(LIBS) 

omp-spmv-vect-row-88.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DRMJR -DBR=8 -DBC=8  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-row-88: omp-spmv-vect-row-88.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-row-88.o $(LIBS) 

omp-spmv-vect-row-48.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DRMJR -DBR=4 -DBC=8  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-row-48: omp-spmv-vect-row-48.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-row-48.o $(LIBS) 


omp-spmv-vect-row-28.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DRMJR -DBR=2 -DBC=8  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-row-28: omp-spmv-vect-row-28.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-row-28.o $(LIBS) 


omp-spmv-vect-row-18.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DRMJR -DBR=1 -DBC=8  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-row-18: omp-spmv-vect-row-18.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-row-18.o $(LIBS) 


omp-spmv-vect-row2-88.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DRMJR2 -DBR=8 -DBC=8  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-row2-88: omp-spmv-vect-row2-88.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-row2-88.o $(LIBS) 


omp-spmv-vect-row2-48.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DRMJR2 -DBR=4 -DBC=8  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-row2-48: omp-spmv-vect-row2-48.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-row2-48.o $(LIBS) 

omp-spmv-vect-row2-28.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DRMJR2 -DBR=2 -DBC=8  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-row2-28: omp-spmv-vect-row2-28.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-row2-28.o $(LIBS) 


omp-spmv-vect-row2-18.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DRMJR2 -DBR=1 -DBC=8  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-row2-18: omp-spmv-vect-row2-18.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-row2-18.o $(LIBS) 


omp-spmv-vect-col-88.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DCMJR -DBR=8 -DBC=8  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-col-88: omp-spmv-vect-col-88.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-col-88.o $(LIBS) 

omp-spmv-vect-col-84.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DCMJR -DBR=8 -DBC=4  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-col-84: omp-spmv-vect-col-84.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-col-84.o $(LIBS) 

omp-spmv-vect-col-82.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DCMJR -DBR=8 -DBC=2  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-col-82: omp-spmv-vect-col-82.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-col-82.o $(LIBS) 


omp-spmv-vect-col-81.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DCMJR -DBR=8 -DBC=1  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-col-81: omp-spmv-vect-col-81.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-col-81.o $(LIBS) 


omp-spmv-vect-swi-88.o: omp-spmv-vect-bcsr.cpp main-spmv.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DSWI -DBR=8 -DBC=8  -c omp-spmv-vect-bcsr.cpp -o $@

omp-spmv-vect-swi-88: omp-spmv-vect-swi-88.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect-swi-88.o $(LIBS) 




omp-spmm: omp-spmm.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmm.o $(LIBS)

omp-spmm-vect-reg.o: omp-spmm-vect.cpp main-spmm.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DREGULAR_STORE  -c omp-spmm-vect.cpp -o $@

omp-spmm-vect-reg: omp-spmm-vect-reg.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmm-vect-reg.o $(LIBS)

omp-spmm-vect-nr.o: omp-spmm-vect.cpp main-spmm.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DNR_STORE  -c omp-spmm-vect.cpp -o $@

omp-spmm-vect-nr: omp-spmm-vect-nr.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmm-vect-nr.o $(LIBS)

omp-spmm-vect-nrngo.o: omp-spmm-vect.cpp main-spmm.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DNRNGO_STORE  -c omp-spmm-vect.cpp -o $@


omp-spmm-vect-nrngo: omp-spmm-vect-nrngo.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmm-vect-nrngo.o $(LIBS)

omp-spmm-vect: omp-spmm-vect.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmm-vect.o $(LIBS)



omp-spmv-vect.o: omp-spmv-vect.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $*.cpp  

omp-spmv-vect: omp-spmv-vect.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-vect.o $(LIBS) 


omp-spmv-pref: omp-spmv-pref.o ulib.o mmio.o
	$(LD) $(LDFLAGS) -o $@ mmio.c ulib.o omp-spmv-pref.o $(LIBS) 


include makefile.in
