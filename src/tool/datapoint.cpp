#include "datapoint.h"
double euclideanDistance(const DataPoint &t1, const DataPoint &t2) {
	double dd = .0;
	for (int d = 0; d < t1.dimensionality(); d++) {
		dd += (t1.x(d) - t2.x(d)) * (t1.x(d) - t2.x(d));
	}
	return dd;
}