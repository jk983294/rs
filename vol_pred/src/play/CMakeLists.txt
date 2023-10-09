add_executable(cn_fut_pred cn_fut_pred.cpp)
target_link_libraries(cn_fut_pred vol_pred_lib zerg_drone_lib frenzy common_3rd_lib pthread rt z)

add_executable(ip_test ip_test.cpp)
target_link_libraries(ip_test pthread rt z)