# Project Description

In urban geography, a city is characterized as a patchwork of intensive land-uses (Storper & Scott, 2016). The function of a particular patch of land and its physical character is socially defined. However, research that rigorously establishes the link between the physical character of a given piece of land with 'use' is lacking despite the proliferation of geospatial data. Linking a city's physical form with its function(s) is the goal of this project. 

## Land-cover
In geographical literature, land-cover is generally understood as the physical composition of the features on the earth's surface (Cihlar & Jansen, 2001). Categories of land-cover generally have a one-one correspondence with physical variables that can be measured and recorded from a sensing instrument. In this case satellite sensors detect the electromagnetic radiation reflected from materials on the earth's surface which are then stored digitally as an array of pixels. Urban land-cover is therefore the materials that are detectable and classifiable in a urban location. Impervious surfaces are characteristic of artificial structures found on landscapes such as cities. This environmental aspect of urban land-cover is one of the main parameters used in classifying urban land-use.

## Land-use
In contrast to land-cover, land-use is a description of how people use the land. This is trickier to measure and classify than land-cover because of the complicating factor of human interpretation of what actually constitutes 'land-use.' Definitions of land-use differ according to location on the earth and depend on who defines 'use' for a particular patch of land (Comber, Fisher and Wadsworth, 2005). Moreover, a patch of land with a particular 'use' is, in many cases, be a composition of different kinds of land-cover (Cihlar & Jansen). Therefore, land-use cannot be directly measured via a scientific instrument unlike land-cover.

## Methodology
The uniqueness of land-use to particular locations can be exploited by combining human expertise with the advantages offered by machine learning algorithms. The assumption behind this method is that land-use can be modelled in terms of environmental variables: vegetation, impervious surfaces and soil (VIS). Urban ecosystems are a composite of these three variables (Ridd, 1991) and therefore can be observed, quantified and measured from satellite images. In this project, the morphology of impervious surfaces will be measured and characterized per land-use category. 

Impervious and pervious (vegetation and soil) surfaces can be encoded into numerical categories and classified using machine learning algorithms. In this project, the VIS can me modelled by taking advantage of the linear correlation of impervious and pervious surfaces in very high resolution (0.5mx0.5m pixels) and medium resolution (30mx30m pixels) satellite images. Impervious surfaces can then be further characterized according to their morphology within arbitrarily defined land-use boundaries and classified into land-use categories. 

## Preliminary Results

![alt_tag](https://github.com/tropicalmentat/land-cover-to-land-use-classification/blob/master/general_workflow.png)

![alt tag](https://github.com/tropicalmentat/land-cover-to-land-use-classification/blob/master/prelim%20land%20cover.png)

![alt tag](https://github.com/tropicalmentat/land-cover-to-land-use-classification/blob/master/land-use%20classification.png)

## Land-use Classification Accuracy Report

|             | agricultural | commercial | industrial | mixed-use | residential |
| --- | --- | --- | --- | --- | --- |
|agricultural      |       4     |      0     |      0     |     0       |     0|
|commercial         |      0      |     1      |     1     |     0       |     2|
|industrial          |     0      |     0      |     1     |     0       |     0|
|mixed-use           |     0       |    0      |     0     |     0       |     2|
|residential       |       0       |    1       |    1     |     1        |    6|

|              |precision  |  recall | f1-score  | support|
|   ---           |---  |  --- | ---  | ---|
|agricultural    |   1.00   |   1.00  |    1.00   |      4|
|  commercial    |   0.25   |   0.50  |    0.33    |     2|
|  industrial   |    1.00   |   0.33   |   0.50   |      3|
|   mixed-use  |     0.00   |   0.00    |  0.00    |     1|
| residential   |    0.67    |  0.60  |    0.63   |     10|
| avg / total    |   0.71   |   0.60   |   0.62    |    20|



### References

Cihlar, J., & Jansen, L. (2001). From Land Cover to Land Use: A Methodology for Efficient Land Use Mapping over Large Areas. The Professional Geographer, 53(2), 275–289. https://doi.org/10.1111/0033-0124.00285

Comber, A., Fisher, P., & Wadsworth, R. (2005). What is land cover? Environment and Planning B: Planning and Design, 32(2), 199–209. https://doi.org/10.1068/b31135

RIDD, M. K. (1995). Exploring a V-I-S (vegetation-impervious surface-soil) model for urban ecosystem analysis through remote sensing: comparative anatomy for cities†. International Journal of Remote Sensing, 16(12), 2165–2185. https://doi.org/10.1080/01431169508954549

Storper, M., & Scott, A. J. (2016). Current debates in urban theory: A critical assessment. Urban Studies, 53(6), 1114–1136. https://doi.org/10.1177/0042098016634002

References on machine learning can be found [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md):




