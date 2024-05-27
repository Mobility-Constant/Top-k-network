# Top-k-network
This repository contains the code for the paper:
> General percolation transition of k-most-frequent destinations network at 130 for urban mobility
> Weiyu Zhang*, Furong Jia*, Jianying Wang, Yu Liu, Gezhi Xiu
> **Abstract:** Urban spatial interactions are a complex aggregation of routine visits and random explorations by individuals. The inherent uncertainty of these random visitations poses significant challenges to understanding urban structures and socioeconomic developments. To capture the core dynamics of urban interaction networks, we analyze the percolation structure of the $k$-most frequented destinations of intracity place-to-place flows from mobile phone data of eight major U.S. cities at a Census Block Group (CBG) level. Our study reveals a consistent percolation transition at $k^* = 130$, a critical threshold for the number of frequently visited destinations necessary to maintain a cohesive urban network. This percolation threshold proves remarkably consistent across diverse urban configurations, sizes, and geographical settings over a 48-month study period, and can largely be interpreted as the joint effect of the emergence of hubness and the level of mixing of residents. Furthermore, we examine the socioeconomic profiles of residents from different origin areas categorized by the fulfillment level of $k^*=130$ principal destinations, revealing a pronounced distinction in the origins' socioeconomic advantages. These insights offer a nuanced understanding of how urban spaces are interconnected and the determinants of travel behavior. Our findings contribute to a deeper comprehension of the structural dynamics that govern urban spatial interactions.

## Code structure

The codes contains:

*   **scripts/plot_flow_network.ipynb** Code used to visualize the original flow and the network of the 130 most frequent destinations, utilized to reproduce Fig. 1A-H.

*   **scripts/plot_PPD_Socioeconomic_relationship.ipynb** Code used to:
    - Plot the spatial distribution of Principal Proportional Destinations (PPD) and reproduce Fig. 3.
    - Conduct correlation analysis of socioeconomic factors with the Proportion of Principal Destinations (PPD) and reproduce Fig. 4.
