add_library(my_lib2
            customMarker.cpp
            HungarianAlg.cpp
            processPointClouds.cpp
            track.cpp)
target_link_libraries(my_lib2 ${catkin_LIBRARIES} ${PCL_LIBRARIES})