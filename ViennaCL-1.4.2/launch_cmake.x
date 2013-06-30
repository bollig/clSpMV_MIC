#!/bin/sh
cmake -DUSE_ICC=on -DBUILD_TESTING=on -DENABLE_OPENCL=on  ..

#-DSETTINGS_LIBRARY=external/settings/libsettings.a 
