cmake_minimum_required( VERSION 2.8 )

# Create Project
project( solution )

add_executable( project 
		main.cpp 
		TimeString.cpp 
		TimeString.h )

# Set StartUp Project (Option)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "project" )

#https://myenigma.hatenablog.com/entry/2015/05/29/220404
include_directories("${CMAKE_SOURCE_DIR}/eigen-3.3.7")