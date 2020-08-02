#include <stdio.h>  
#include <stdlib.h>
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <geometry_msgs/PoseArray.h>

//Client definition
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

//Action Feedback and Result
// typedef boost::shared_ptr< ::move_base_msgs::MoveBaseActionFeedback const > MoveBaseActionFeedbackConstPtr;
typedef boost::shared_ptr< ::move_base_msgs::MoveBaseFeedback const > MoveBaseActionFeedbackConstPtr;
// typedef boost::shared_ptr< ::move_base_msgs::MoveBaseActionResult const > MoveBaseActionResultConstPtr;
typedef boost::shared_ptr< ::move_base_msgs::MoveBaseResult const > MoveBaseActionResultConstPtr;

//Node Feedback and Result
typedef boost::shared_ptr< ::move_base_msgs::MoveBaseResult const > MoveBaseResultConstPtr;
typedef boost::shared_ptr< ::move_base_msgs::MoveBaseFeedback const > MoveBaseFeedbackConstPtr;

class GoalSender {
 private:
  // Subscriber
  ros::Subscriber goals_sub_;

  // Params
  std::string goals_topic_;
  std::string node_name;

  // Client
  MoveBaseClient MB_client;

  // Goal sent to Move Base
  move_base_msgs::MoveBaseActionGoal action_goal;
  move_base_msgs::MoveBaseGoal goal;

  // Goals Array received from other node
  geometry_msgs::Pose waypoint;

  // Number of current goal in the waypoint array
  int index = 0;

 public:
  // ROS Handle
  ros::NodeHandle nh;
  GoalSender()
      : MB_client("move_base", true)

  {
    // Get Node name
    node_name = ros::this_node::getName();

    // Wait for server to be online
    while (!MB_client.waitForServer(ros::Duration(5.0))) {
      ROS_INFO("%s: Waiting for move_base action server start",
               node_name.c_str());
    }
    ROS_INFO("%s: Action sever started", node_name.c_str());

    // Start private ros node handle to retrieve parameter
    // a ~ puts a topic or a param in the nodes private namespace
    ros::NodeHandle nh_p("~");

    if (nh_p.getParam("goals_topic", goals_topic_)) {
      ROS_INFO("%s: Fetching goals array on %s", node_name.c_str(),
               goals_topic_.c_str());
    } else {
      ROS_WARN("%s: No goals topic given, Defaulting to input_goal",
               node_name.c_str());
      goals_topic_ = "input_goal";
    }

    goals_sub_ =
        nh.subscribe(goals_topic_, 5, &GoalSender::goals_registering, this);
  }

  bool checkWaypoints(geometry_msgs::Pose waypoint) {
    // TODO
    return true;
  }

  // Callback function to goals array
  void goals_registering(const geometry_msgs::Pose msg) {
    if (!checkWaypoints(msg)) {
      ROS_WARN(
          "%s: Received waypoint were corrupted or undoable. No goal will "
          "be sent",
          node_name.c_str());
    } else {
      ROS_INFO("%s: Received waypoint were correct. Proceeding to goal 1",
               node_name.c_str());
      index = 0;
      sendGoald(msg, index++);
    }
  }

  // To send Goal to navigation stack
  void sendGoald(geometry_msgs::Pose goal_pose, int index) {
    ROS_DEBUG("%s: A new goal pose will be sent over the path planning node",
              node_name.c_str());
    ros::Time now = ros::Time::now();

    /* Fill up action goal info */
    // ID (stamp and id)
    action_goal.goal_id.stamp = now;
    action_goal.goal_id.id = std::to_string(index);

    // Goal (geometry_msgs/PoseStamped)
    action_goal.goal.target_pose.header.seq = index;
    action_goal.goal.target_pose.header.frame_id = "base_link";
    action_goal.goal.target_pose.header.stamp = now;
    action_goal.goal.target_pose.pose.position.x = goal_pose.position.x;
    action_goal.goal.target_pose.pose.position.y = goal_pose.position.y;
    action_goal.goal.target_pose.pose.position.z = goal_pose.position.z;
    action_goal.goal.target_pose.pose.orientation.w = goal_pose.orientation.w;

    // Header
    action_goal.header.stamp = now;
    action_goal.header.frame_id = "base_link";
    goal = action_goal.goal;
    MB_client.sendGoal(goal, boost::bind(&GoalSender::done, this, _1, _2),
                       boost::bind(&GoalSender::activ, this),
                       boost::bind(&GoalSender::feedback, this, _1));
    // MB_client.sendGoal(goal);

                    
            
  }

  void done(const actionlib::SimpleClientGoalState& state,
            const MoveBaseActionResultConstPtr& result) {
    ROS_INFO("%s: Finished in state [%s]", node_name.c_str(),
             state.toString().c_str());
    /***
     * TO DO: if goal reached, send the next one
     ***/
  }

  void activ(void) {
    /***
     * TO DO
     ***/
  }

  void feedback(const MoveBaseActionFeedbackConstPtr& feedback) {
    /***
     * TO DO
     ***/
    ROS_INFO("%s: Feed back received", node_name.c_str());
    show_pose(feedback->base_position);

  }

  void show_pose(geometry_msgs::PoseStamped pose) {
    ROS_INFO("pose.header.seq %d", pose.header.seq);
    ROS_INFO("pose.header.frame_id %s", pose.header.frame_id.c_str());
    ROS_INFO("pose.pose.position.x %.2f", pose.pose.position.x);
    ROS_INFO("pose.pose.position.y %.2f", pose.pose.position.y);
    ROS_INFO("pose.pose.position.z %.2f", pose.pose.position.z);
    ROS_INFO("pose.pose.orientation.w %.2f", pose.pose.orientation.w);
    
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "navigatation_goals");
  GoalSender goalSender;
  ros::spin();
  return 0;
}