#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <ros/ros.h>

using namespace std;

int main(int argc, char **argv) {
  ros::init(argc, argv, "input_goal");

  ros::NodeHandle nh;

  ros::Publisher goal_pub =
      nh.advertise<geometry_msgs::Pose>("input_goal", 1000);

  ros::Rate loop_rate(0.5);
  int index = 0;
  while (ros::ok()) {
    geometry_msgs::Pose goal;
    cout << "enter point\n";
    cout << "x: ";
    cin >> goal.position.x;
    cout << "y: ";
    cin >> goal.position.y;
    cout << "z: ";
    cin >> goal.position.z;

    cout << "enter orientation\n";
    cout << "w: ";
    cin >> goal.orientation.w;
    goal_pub.publish(goal);
  }
}
