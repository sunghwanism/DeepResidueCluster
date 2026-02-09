import datetime
import random
import sys
import networkx as nx
import pickle



'''
Code: https://github.com/murselTasginBoun/CDBN
--------------------------------------------------------------------------
[Paper]
Tasgin, M. and H. O. Bingol, “Community detection using boundary nodes in
complex networks”, Physica A: Statistical Mechanics and its Applications
Vol.513,pp.315–324,2019
--------------------------------------------------------------------------
'''

def now(format=None):
    if format:
        return datetime.datetime.now().strftime(format)
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class CommunityDetection:
    def __init__(self, unweighted):
        self.CDwBN_method = 2
        self.benefitGroupBy = 2
        self.G = nx.Graph()
        self.unweighted = unweighted

    def read_pajek_manual(self, file_path):
        G = nx.Graph()
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        vertices_line_index = -1
        edges_line_index = -1
        for i, line in enumerate(lines):
            if line.lower().startswith("*vertices"):
                vertices_line_index = i
            elif line.lower().startswith("*edges"):
                edges_line_index = i
                break
        
        if vertices_line_index != -1:
            for i in range(vertices_line_index + 1, edges_line_index if edges_line_index != -1 else len(lines)):
                parts = lines[i].split()
                if len(parts) > 0:
                    G.add_node(parts[0])

        if edges_line_index != -1:
            for i in range(edges_line_index + 1, len(lines)):
                parts = lines[i].split()
                if len(parts) > 1:
                    u, v = parts[0], parts[1]
                    weight = float(parts[2]) if len(parts) > 2 else 1.0
                    G.add_edge(u, v, weight=weight)

        return G

    def read_dat_manual(self, file_path):
        G = nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=str, data=[('weight', float)])
        return G

    def communityDetection(self, fileN):
        self.inputFileName = fileN
        if isinstance(fileN, nx.Graph):
            self.G = fileN
        elif self.inputFileName.endswith(".gml"):
            self.G = nx.read_gml(self.inputFileName, label='id')
        elif self.inputFileName.endswith(".txt") or self.inputFileName.endswith(".net"):
            self.G = self.read_pajek_manual(self.inputFileName)
        elif self.inputFileName.endswith(".dat"):
            self.G = self.read_dat_manual(self.inputFileName)
        elif self.inputFileName.endswith(".pkl"):
            with open(self.inputFileName, 'rb') as f:
                self.G = pickle.load(f)
        

        if self.CDwBN_method == 1:
            print(now() + "         --> CDwBN with Individual approach is running")
            self.benefitGroupBy = 1
        elif self.CDwBN_method == 2:
            print(now() + "         --> CDwBN with Group approach is running")
            self.benefitGroupBy = 2
        
        self.communityDetectionBoundaryNodes(self.G)
        
        CommunityCount = {}
        nodes = list(self.G.nodes())
        for node in nodes:
            if 'communityID' not in self.G.nodes[node]:
                self.G.nodes[node]['communityID'] = node
            self.G.nodes[node]['communityArc'] = self.G.nodes[node]['communityID']
            communityID = self.findCommunityOfNode(self.G, node)
            self.G.nodes[node]['communityID'] = communityID
            CommunityCount[communityID] = communityID
            
        # filename = self.inputFileName[:self.inputFileName.rfind(".")] + ".communities.txt"
        # self.printCMTYfile(self.G, filename)
        # return filename
        return self.G

    def communityDetectionBoundaryNodes(self, G):
        print(now() + " FUNCTION communityDetectionBoundaryNodes started---------------")
        
        boundaryNodeStack = []
        benefitScores = {}
        communityMembers = {}
        
        nodes = list(G.nodes())
        for node in nodes:
            G.nodes[node]['communityID'] = node
            G.nodes[node]['boundaryState'] = 0
            G.nodes[node]['boundaryNodeStateCounter'] = 0
            G.nodes[node]['CIDchangeCount'] = 0
            G.nodes[node]['eventLog'] = {}

        self.calculateEdgeWeights(G, self.unweighted)
        
        for node in nodes:
            maxW = 0
            idx = node
            for neighbor in G.neighbors(node):
                w = G[node][neighbor]['weight']
                if w > maxW:
                    maxW = w
                    idx = neighbor
            G.nodes[node]['communityID'] = G.nodes[idx]['communityID']
            
        print(now() + "         --> boundaryNodes initialization done!")
        
        for node in nodes:
            for neighbor in G.neighbors(node):
                if G.nodes[node]['communityID'] != G.nodes[neighbor]['communityID'] and G.nodes[node]['boundaryState'] == 0:
                    boundaryNodeStack.append(node)
                    G.nodes[node]['boundaryNodeStateCounter'] += 1
                    G.nodes[node]['boundaryState'] = 1
                    break
                    
        print(now() + "         --> Initial boundary stack size=" + str(len(boundaryNodeStack)))
        
        loopCount = 0
        messageFreq = 5000
        while boundaryNodeStack:
            loopCount += 1
            
            randIndex = random.randint(0, len(boundaryNodeStack) - 1)
            currentID = boundaryNodeStack.pop(randIndex)
            
            G.nodes[currentID]['boundaryState'] = 0
            if loopCount % messageFreq == 0:
                print(f"{now()}         --> communityDetection counter:{loopCount} boundaryNodeStack.size:{len(boundaryNodeStack)} current node G.degree[{currentID}]:{G.degree(currentID)}")

            benefitScores.clear()
            communityMembers.clear()

            for neighbor in G.neighbors(currentID):
                commID = G.nodes[neighbor]['communityID']
                benefit = G[currentID][neighbor]['weight']

                if self.benefitGroupBy == 1:
                    benefitScores[neighbor] = benefit
                elif self.benefitGroupBy == 2:
                    if commID in benefitScores:
                        benefitScores[commID] += benefit
                    else:
                        benefitScores[commID] = benefit
                    
                    if commID in communityMembers:
                        communityMembers[commID] += 1
                    else:
                        communityMembers[commID] = 1

            highestScore = -999999
            highestCommID = -1
            for IDx, currentScore in benefitScores.items():
                if currentScore > highestScore:
                    highestScore = currentScore
                    highestCommID = IDx
            
            finalScores = {}
            for ID, scoreVal in benefitScores.items():
                if scoreVal == highestScore:
                    finalScores[ID] = scoreVal
            
            currentCommIDofNode = G.nodes[currentID]['communityID']
            
            if len(finalScores) > 0:
                if currentCommIDofNode not in finalScores:
                    if finalScores:
                        randomIDx = random.randint(0, len(finalScores) - 1)
                        ID = list(finalScores.keys())[randomIDx]
                        highestCommID = ID
                else:
                    highestCommID = currentCommIDofNode
            
            currentNodeCommID = G.nodes[currentID]['communityID']
            
            if currentNodeCommID != highestCommID and highestCommID != -1:
                oldCommID = G.nodes[currentID]['communityID']
                G.nodes[currentID]['communityID'] = highestCommID
                G.nodes[currentID]['CIDchangeCount'] += 1
                tmpHistory = f"Comm:{oldCommID}>{highestCommID};changeCount;{G.nodes[currentID]['CIDchangeCount']}"
                G.nodes[currentID]['eventLog'][len(G.nodes[currentID]['eventLog'])] = tmpHistory
                
                for neighbor in G.neighbors(currentID):
                    nodeCommID = G.nodes[currentID]['communityID']
                    neighbourCommID = G.nodes[neighbor]['communityID']
                    
                    if nodeCommID != neighbourCommID and G.nodes[neighbor]['boundaryState'] == 0:
                        boundaryNodeStack.append(neighbor)
                        G.nodes[neighbor]['boundaryNodeStateCounter'] += 1
                        G.nodes[neighbor]['boundaryState'] = 1
                        
        print(now() + " FUNCTION-communityDetectionBoundaryNodes ended---------------")

    def findCommunityOfNode(self, G, node):
        linklist = {node: node}
        if 'communityID' not in G.nodes[node]:
            G.nodes[node]['communityID'] = node

        transitID = G.nodes[node]['communityID']
        while transitID not in linklist:
            linklist[transitID] = transitID
            if 'communityID' not in G.nodes[transitID]:
                G.nodes[transitID]['communityID'] = transitID
            transitID = G.nodes[transitID]['communityID']
            
        for idx in linklist:
            G.nodes[idx]['communityID'] = transitID
        return transitID

    def calculateEdgeWeights(self, G, unweighted=False):
        print(now() + " FUNCTION-calculateEdgeWeights started---------------")
        if unweighted:
            for u, v in G.edges():
                G[u][v]['weight'] = 1.0
        else:
            loopCounter = 0
            printFreq = 50000
            for u, v in G.edges():
                common_neighbors = len(list(nx.common_neighbors(G, u, v)))
                G[u][v]['weight'] = common_neighbors
                loopCounter += 1
                if loopCounter % printFreq == 0:
                    print(now() + "         --> calculateEdgeWeight counter:" + str(loopCounter))
        print(now() + " FUNCTION-calculateEdgeWeights ended---------------")

    def printCMTYfile(self, G, outputFileName):
        print(now() + " FUNCTION-printCMTYfile started------------------------------")
        print(now() + "         --> OutputFile:" + outputFileName)
        
        communityMap = {}
        for node in G.nodes():
            communityID = G.nodes[node]['communityID']
            if communityID not in communityMap:
                communityMap[communityID] = ""
            communityMap[communityID] += str(node) + " "
            
        print(now() + "         -----------------------------------------")
        print(now() + "         --> Number of nodes      :" + str(G.number_of_nodes()))
        print(now() + "         --> Number of edges      :" + str(G.number_of_edges()))
        print(now() + "         --> Number of communities:" + str(len(communityMap)))
        print(now() + "         -----------------------------------------")
        
        with open(outputFileName, "w") as writer:
            for communityID, memberLIST in communityMap.items():
                writer.write(memberLIST.strip() + "\n")
                
        print(now() + "         --> Communities are written to .cmty file:" + outputFileName)
        print(now() + " FUNCTION-printCMTYfile ended------------------------------")

if __name__ == "__main__":
    headerMsg = " ------Community detection using boundary nodes ------------------------------\n"
    headerMsg += "|>> Citation info >>:                                                        |\n"
    citationMsg = "| Tasgin, M. and H. O. Bingol, “Community detection using boundary nodes in  |\n"
    citationMsg += "| complex networks”, Physica A: Statistical Mechanics and its Applications   |\n"
    citationMsg += "| Vol.513,pp.315–324,2019                                                    |\n"
    citationMsg += "| http://www.sciencedirect.com/science/article/pii/S0378437118311658         |\n"
    citationMsg += " ----------------------------------------------------------------------------\n"
    
    print(headerMsg)
    print(citationMsg)
    
    if len(sys.argv) < 2:
        print("Usage: python community_detection.py inputfilename [option]")
        print("Option:1           - CDwBN - Community detection using boundary nodes - Individual approach")
        print("Option:2 (default) - CDwBN - Community detection using boundary nodes - Group approach")
        sys.exit(1)
        
    inputfile = sys.argv[1]
    option = 2
    if len(sys.argv) > 2:
        option = int(sys.argv[2])
        
    if option not in [1, 2]:
        print("Invalid option given!")
        print("Usage: python community_detection.py inputfilename [option]")
        print("Option:1           - CDwBN - Community detection using boundary nodes - Individual approach")
        print("Option:2 (default) - CDwBN - Community detection using boundary nodes - Group approach")
        sys.exit(1)
        
    givenOption = "Group approach"
    if option == 1:
        givenOption = "Individual approach"
        
    print(f"{now()}         --> Working on inputfile:{inputfile} using CDwBN with the given option:{givenOption}")
    
    C1 = CommunityDetection()
    C1.CDwBN_method = option
    C1.communityDetection(inputfile)