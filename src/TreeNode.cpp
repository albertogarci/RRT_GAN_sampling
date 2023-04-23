#include <string>
#include <vector>
#include <iostream>
#include <geometry_msgs/PoseStamped.h>
#include "AROB_lab4/TreeNode.h"

/*
TreeNode::TreeNode()
{
    std::cout << "Constructor" << std::endl;
    parent = NULL;
}
*/
TreeNode::TreeNode(std::vector<int> point_)
{
    // std::cout << "Constructor" << std::endl;
    // std::cout << "point_:" << point_[0] << ", " << point_[1] << std::endl;
    point = point_;
    parent = NULL;
}

TreeNode::~TreeNode()
{
    TreeNode *child = NULL;
    TreeNode *parent = this;
    // std::cout << "Destructor" << std::endl;
    for (int it = 0; it < parent->childrenNumber(); it++)
    {
        child = parent->getChild(it);
        if (child->hasChildren())
        {
            child->~TreeNode();
        }
        else
        {
            child->point.clear();
            child->children.clear();
            child->parent = NULL;
            //delete child;
        }
    }
    this->point.clear();
    this->children.clear();
    this->parent = NULL;
}

bool TreeNode::hasChildren()
{
    if (children.size() > 0)
        return true;
    else
        return false;
}

int TreeNode::countNodesRec(TreeNode *root, int &count)
{
    TreeNode *parent = root;
    TreeNode *child = NULL;

    for (int it = 0; it < parent->childrenNumber(); it++)
    {
        child = parent->getChild(it);
        count++;
        if (child->childrenNumber() > 0)
        {
            countNodesRec(child, count);
        }
    }

    return count;
}

void TreeNode::appendChild(TreeNode *child)
{
    child->setParent(this);
    children.push_back(child);
}

void TreeNode::setParent(TreeNode *theParent)
{
    parent = theParent;
}

bool TreeNode::hasParent()
{
    if (parent != NULL)
        return true;
    else
        return false;
}

/*
TreeNode *TreeNode::TreeNode()
{
    return parent;
}
*/
TreeNode *TreeNode::getChild(int pos)
{
    if (children.size() < pos)
        return NULL;
    else
        return children[pos];
}

int TreeNode::childrenNumber()
{
    return children.size();
}

std::vector<int> TreeNode::getNode()
{
    return point;
}

void TreeNode::printNode()
{
    std::cout << "Node: (" << point[0] << "," << point[1] << ")." << std::endl;
}

void TreeNode::printTree()
{
    TreeNode *child = NULL;

    std::cout << "Parent node: (" << point[0] << "," << point[1] << ")." << std::endl;

    for (int it = 0; it < children.size(); it++)
    {
        std::cout << "    Child node: (" << children[it]->point[0] << "," << children[it]->point[1] << ")." << std::endl;
    }
    for (int it = 0; it < children.size(); it++)
    {
        children[it]->printTree();
    }
}

TreeNode *TreeNode::nearNode(TreeNode *node1, TreeNode *node2)
{
    std::vector<int> pos1 = node1->getNode();
    std::vector<int> pos2 = node2->getNode();
    int distance1 = sqrt((pos1[0] - point[0]) * (pos1[0] - point[0]) + (pos1[1] - point[1]) * (pos1[1] - point[1]));
    int distance2 = sqrt((pos2[0] - point[0]) * (pos2[0] - point[0]) + (pos2[1] - point[1]) * (pos2[1] - point[1]));
    if (distance1 < distance2)
        return node1;
    else
        return node2;
}

TreeNode *TreeNode::neast(TreeNode *root)
{

    TreeNode *parent = root;
    TreeNode *child = NULL;
    TreeNode *shortest = parent;

    for (int it = 0; it < parent->childrenNumber(); it++)
    {
        child = parent->getChild(it);
        shortest = nearNode(shortest, neast(child));
    }
    return shortest;
}
/*
TreeNode *TreeNode::get_nearest(TreeNode *root, TreeNode *node)
{

    TreeNode *parent = root;
    TreeNode *child = NULL;
    TreeNode *shortest = node;

    for (int it = 0; it < parent->childrenNumber(); it++)
    {
        child = parent->getChild(it);
        shortest = nearNode(shortest, neast(child));
    }
    return shortest;
}
*/
TreeNode *TreeNode::GenerateRandomNode(std::uniform_int_distribution<std::mt19937::result_type> &distr, 
                                std::uniform_int_distribution<std::mt19937::result_type> &distr2, std::mt19937 &eng)
{
    int x = distr(eng);
    int y = distr2(eng);
    //std::cout << x << ", " << y << std::endl;
    std::vector<int> point{(int)x, (int)y};
    return new TreeNode(point);
}

std::vector<std::vector<int>> TreeNode::returnSolution()
{

    std::vector<std::vector<int>> solution;
    TreeNode *node = this;
    //	std::cout << "Start returning the solution" << std::endl;
    while (node->hasParent())
    {
        solution.push_back(node->getNode());
        if (node->hasParent())
            node = node->parent;
        //		node->printNode();
    };
    //	std::cout << "Finish returning the solution" << std::endl;
    return solution;
}
