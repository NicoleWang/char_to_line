# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wangyuzhuo/wangyuzhuo/char_to_line

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wangyuzhuo/wangyuzhuo/char_to_line/build

# Include any dependencies generated for this target.
include CMakeFiles/DetectChar.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DetectChar.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DetectChar.dir/flags.make

CMakeFiles/DetectChar.dir/test/test_so.cpp.o: CMakeFiles/DetectChar.dir/flags.make
CMakeFiles/DetectChar.dir/test/test_so.cpp.o: ../test/test_so.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/wangyuzhuo/wangyuzhuo/char_to_line/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/DetectChar.dir/test/test_so.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DetectChar.dir/test/test_so.cpp.o -c /home/wangyuzhuo/wangyuzhuo/char_to_line/test/test_so.cpp

CMakeFiles/DetectChar.dir/test/test_so.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DetectChar.dir/test/test_so.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/wangyuzhuo/wangyuzhuo/char_to_line/test/test_so.cpp > CMakeFiles/DetectChar.dir/test/test_so.cpp.i

CMakeFiles/DetectChar.dir/test/test_so.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DetectChar.dir/test/test_so.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/wangyuzhuo/wangyuzhuo/char_to_line/test/test_so.cpp -o CMakeFiles/DetectChar.dir/test/test_so.cpp.s

CMakeFiles/DetectChar.dir/test/test_so.cpp.o.requires:
.PHONY : CMakeFiles/DetectChar.dir/test/test_so.cpp.o.requires

CMakeFiles/DetectChar.dir/test/test_so.cpp.o.provides: CMakeFiles/DetectChar.dir/test/test_so.cpp.o.requires
	$(MAKE) -f CMakeFiles/DetectChar.dir/build.make CMakeFiles/DetectChar.dir/test/test_so.cpp.o.provides.build
.PHONY : CMakeFiles/DetectChar.dir/test/test_so.cpp.o.provides

CMakeFiles/DetectChar.dir/test/test_so.cpp.o.provides.build: CMakeFiles/DetectChar.dir/test/test_so.cpp.o

# Object files for target DetectChar
DetectChar_OBJECTS = \
"CMakeFiles/DetectChar.dir/test/test_so.cpp.o"

# External object files for target DetectChar
DetectChar_EXTERNAL_OBJECTS =

DetectChar: CMakeFiles/DetectChar.dir/test/test_so.cpp.o
DetectChar: CMakeFiles/DetectChar.dir/build.make
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
DetectChar: /usr/lib/x86_64-linux-gnu/libboost_system.so
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.9
DetectChar: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9
DetectChar: CMakeFiles/DetectChar.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable DetectChar"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DetectChar.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DetectChar.dir/build: DetectChar
.PHONY : CMakeFiles/DetectChar.dir/build

CMakeFiles/DetectChar.dir/requires: CMakeFiles/DetectChar.dir/test/test_so.cpp.o.requires
.PHONY : CMakeFiles/DetectChar.dir/requires

CMakeFiles/DetectChar.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DetectChar.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DetectChar.dir/clean

CMakeFiles/DetectChar.dir/depend:
	cd /home/wangyuzhuo/wangyuzhuo/char_to_line/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wangyuzhuo/wangyuzhuo/char_to_line /home/wangyuzhuo/wangyuzhuo/char_to_line /home/wangyuzhuo/wangyuzhuo/char_to_line/build /home/wangyuzhuo/wangyuzhuo/char_to_line/build /home/wangyuzhuo/wangyuzhuo/char_to_line/build/CMakeFiles/DetectChar.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DetectChar.dir/depend
