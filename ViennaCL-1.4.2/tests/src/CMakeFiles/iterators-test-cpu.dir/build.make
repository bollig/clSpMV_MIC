# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/local/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests/src

# Include any dependencies generated for this target.
include CMakeFiles/iterators-test-cpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/iterators-test-cpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/iterators-test-cpu.dir/flags.make

CMakeFiles/iterators-test-cpu.dir/iterators.o: CMakeFiles/iterators-test-cpu.dir/flags.make
CMakeFiles/iterators-test-cpu.dir/iterators.o: iterators.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests/src/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/iterators-test-cpu.dir/iterators.o"
	/opt/intel/composer_xe_2013.3.163/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/iterators-test-cpu.dir/iterators.o -c /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests/src/iterators.cpp

CMakeFiles/iterators-test-cpu.dir/iterators.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/iterators-test-cpu.dir/iterators.i"
	/opt/intel/composer_xe_2013.3.163/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests/src/iterators.cpp > CMakeFiles/iterators-test-cpu.dir/iterators.i

CMakeFiles/iterators-test-cpu.dir/iterators.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/iterators-test-cpu.dir/iterators.s"
	/opt/intel/composer_xe_2013.3.163/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests/src/iterators.cpp -o CMakeFiles/iterators-test-cpu.dir/iterators.s

CMakeFiles/iterators-test-cpu.dir/iterators.o.requires:
.PHONY : CMakeFiles/iterators-test-cpu.dir/iterators.o.requires

CMakeFiles/iterators-test-cpu.dir/iterators.o.provides: CMakeFiles/iterators-test-cpu.dir/iterators.o.requires
	$(MAKE) -f CMakeFiles/iterators-test-cpu.dir/build.make CMakeFiles/iterators-test-cpu.dir/iterators.o.provides.build
.PHONY : CMakeFiles/iterators-test-cpu.dir/iterators.o.provides

CMakeFiles/iterators-test-cpu.dir/iterators.o.provides.build: CMakeFiles/iterators-test-cpu.dir/iterators.o

# Object files for target iterators-test-cpu
iterators__test__cpu_OBJECTS = \
"CMakeFiles/iterators-test-cpu.dir/iterators.o"

# External object files for target iterators-test-cpu
iterators__test__cpu_EXTERNAL_OBJECTS =

iterators-test-cpu: CMakeFiles/iterators-test-cpu.dir/iterators.o
iterators-test-cpu: CMakeFiles/iterators-test-cpu.dir/build.make
iterators-test-cpu: CMakeFiles/iterators-test-cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable iterators-test-cpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/iterators-test-cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/iterators-test-cpu.dir/build: iterators-test-cpu
.PHONY : CMakeFiles/iterators-test-cpu.dir/build

CMakeFiles/iterators-test-cpu.dir/requires: CMakeFiles/iterators-test-cpu.dir/iterators.o.requires
.PHONY : CMakeFiles/iterators-test-cpu.dir/requires

CMakeFiles/iterators-test-cpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/iterators-test-cpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/iterators-test-cpu.dir/clean

CMakeFiles/iterators-test-cpu.dir/depend:
	cd /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests/src /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests/src /mnt/global/LCSE/gerlebacher/src/ViennaCL-1.4.2/tests/src/CMakeFiles/iterators-test-cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/iterators-test-cpu.dir/depend
