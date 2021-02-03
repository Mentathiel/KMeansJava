import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;

public class KMeans {

    static final Double PRECISION = 0.0;

    /* K-Means++ implementation, initializes K centroids from data */
    static LinkedList<HashMap<String, Double>> kmeanspp(DataSet data, int K){
        LinkedList<HashMap<String,Double>> centroids = new LinkedList<>();

        centroids.add(data.randomFromDataSet());

        for(int i=1; i<K; i++){
            centroids.add(data.calculateWeighedCentroid());
        }

        return centroids;
    }

    /* K-Means itself, it takes a dataset and a number K and adds class numbers
    * to records in the dataset */
    static void kmeans(DataSet data, int K){
        // select K initial centroids
        LinkedList<HashMap<String,Double>> centroids = kmeanspp(data, K);

        // initialize Sum of Squared Errors to max, we'll lower it at each iteration
        Double SSE = Double.MAX_VALUE;

        while (true) {

            // assign observations to centroids

            var records = data.getRecords();

            // for each record
            for(var record : records){
                Double minDist = Double.MAX_VALUE;
                // find the centroid at a minimum distance from it and add the record to its cluster
                for(int i=0; i<centroids.size(); i++){
                    Double dist = DataSet.euclideanDistance(centroids.get(i), record.getRecord());
                    if(dist<minDist){
                        minDist = dist;
                        record.setClusterNo(i);
                    }
                }

            }

            // recompute centroids according to new cluster assignments
            centroids = data.recomputeCentroids(K);

            // exit condition, SSE changed less than PRECISION parameter
            Double newSSE = data.calculateTotalSSE(centroids);
            if(SSE-newSSE <= PRECISION){
                break;
            }
            SSE = newSSE;
        }
    }

    public static void main(String[] args) {
        try {
            // read data
            DataSet data = new DataSet("files/sample.csv");

            // remove prior classification attr if it exists (input any irrelevant attributes)
            data.removeAttr("Class");

            // cluster
            kmeans(data, 2);

            // output into a csv
            data.createCsvOutput("files/sampleClustered.csv");

        } catch (IOException e){
            e.printStackTrace();
        }
    }

}