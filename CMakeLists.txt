cmake_minimum_required (VERSION 3.15)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
  cmake_policy(SET CMP0028 NEW)
endif()

find_package(junosw REQUIRED)
find_package(ROOT REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

include_directories(${ROOT_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
link_directories(${ROOT_LIBRARY_DIRS} ${HDF5_LIBRARY_DIRS})

add_executable(read Reader.cpp)

set_target_properties(read PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES)

target_link_libraries(read ${ROOT_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})

if (junosw_FOUND)
    target_link_libraries(read
                            junosw::EDMUtil 
                            junosw::EvtNavigator
                            junosw::ElecEvent
                            junosw::GenEvent
                            junosw::SimEvent
                            junosw::CalibEvent
                            junosw::Identifier
                            junosw::GeometryLib
                          )
endif()

set_target_properties(read PROPERTIES RUNTIME_OUTPUT_DIRECTORY "/pbs/home/l/ljones/LEMO")
