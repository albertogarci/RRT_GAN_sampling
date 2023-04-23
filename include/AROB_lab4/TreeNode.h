#ifndef TreeNode_H
#define TreeNode_H

#include <string>
#include <vector>
#include <geometry_msgs/PoseStamped.h>
#include <random>

class TreeNode
{
    private:
        std::vector <int> point;
        TreeNode *parent;
        std::vector<TreeNode *> children;


    public:

        TreeNode();
		TreeNode(std::vector <int> point_);
		~TreeNode();
        int countNodesRec(TreeNode *root, int& count);

		bool hasChildren();
        void appendChild(TreeNode *child);
        void setParent(TreeNode *parent);

        bool hasChildren() const { return children.size() > 0; }
        bool hasParent();

        TreeNode* getParent();
        TreeNode* getChild(int pos);

        int childrenNumber();


		std::vector <int> getNode();
		void printNode();
		void printTree();

		std::vector <std::vector <int>> returnSolution();
	    static TreeNode* GenerateRandomNode(std::uniform_int_distribution<std::mt19937::result_type> &distr, 
                                std::uniform_int_distribution<std::mt19937::result_type> &distr2, std::mt19937 &eng);
        TreeNode* get_nearest(TreeNode *root, TreeNode *node);

		TreeNode* nearNode(TreeNode* node1, TreeNode* node2);
		TreeNode* neast(TreeNode *root);
};

#endif
