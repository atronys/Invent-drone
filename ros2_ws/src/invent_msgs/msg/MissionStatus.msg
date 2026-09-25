# Статус узла миссии
std_msgs/Header header
string state                      # WAIT, ARMING, FOLLOW, LAND, DONE, ABORT
uint32 sortie                     # номер вылета в плане
uint32 waypoint_index
uint32 waypoint_count
string[] targets                  # адреса, которые должны быть в кадре на текущей точке
