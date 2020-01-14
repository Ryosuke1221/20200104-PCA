cmake_minimum_required( VERSION 2.8 )

# Create Project
project( solution )

add_executable( project 
		main.cpp 
		TimeString.cpp 
		TimeString.h )

# Set StartUp Project (Option)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "project" )

include_directories("${CMAKE_SOURCE_DIR}/eigen-3.3.7")