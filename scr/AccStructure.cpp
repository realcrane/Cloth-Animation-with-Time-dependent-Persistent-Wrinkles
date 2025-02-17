#include "AccStructure.h"

void collect_leaves(DeformBVHNode* node, std::vector<DeformBVHNode*>& leaves)
{
    if (node->isLeaf()) {
        size_t f = node->getFace()->index;
        if (f >= leaves.size())
            leaves.resize(f + 1);
        leaves[f] = node;
    }
    else {
        collect_leaves(node->getLeftChild(), leaves);
        collect_leaves(node->getRightChild(), leaves);
    }
}

AccelStruct::AccelStruct(const Mesh& mesh, bool is_ccd): tree(const_cast<Mesh&>(mesh), is_ccd)
{
	root = tree.root;

	leaves = std::vector<DeformBVHNode*>(mesh.faces.size());

	if (root != nullptr) {
		collect_leaves(root, leaves);
	}
}

std::vector<AccelStruct*> create_accel_structs(const std::vector<Mesh*>& meshes, bool ccd)
{
    std::vector<AccelStruct*> accs(meshes.size());

    for (int m = 0; m < meshes.size(); m++) {
        accs[m] = new AccelStruct(*meshes[m], ccd);
    }

    return accs;
}

void update_accel_struct(AccelStruct& acc)
{
    if (acc.root)
    {
        acc.tree.refit();
    }
}

void destroy_accel_structs(std::vector<AccelStruct*>& accs)
{
    for (int a = 0; a < accs.size(); a++)
    {
        delete accs[a];
    }
}
