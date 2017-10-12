package org.apache.ignite.ml.encog;

import java.util.Arrays;
import java.util.List;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.IgniteDataStreamer;
import org.apache.ignite.cache.CacheAtomicityMode;
import org.apache.ignite.cache.CacheMode;
import org.apache.ignite.cache.CacheWriteSynchronizationMode;
import org.apache.ignite.configuration.CacheConfiguration;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;

public class SamplesCache {
    public static final String CACHE_NAME = "WAV_EXAMPLE_CACHE";

    /**
     * Load wav into cache.
     *
     * @param wav Wav.
     * @param histDepth History depth.
     */
    public static void loadIntoCache(List<double[]> wav, int histDepth, int maxSamples, int stepSize, String cacheName, Ignite ignite) {
        IgniteCache<Integer, MLDataPair> cache = getOrCreate(ignite);
        System.out.println("initial cache size is " + cache.size());

        try (IgniteDataStreamer<Integer, MLDataPair> stmr = ignite.dataStreamer(cacheName)) {
            stmr.allowOverwrite(true);
            // Stream entries.
            int samplesCnt = wav.size();
            System.out.println("File contains " + samplesCnt + " samples...");
            for (int i = histDepth; i < samplesCnt - 1 && (i - histDepth) < maxSamples; i++) {
                if ((i - histDepth) % stepSize != 0)
                    continue;
                // The mean is calculated inefficient
                BasicMLData dataSetEntry = new BasicMLData(wav.subList(i - histDepth, i).stream().map(doubles ->
                    (Arrays.stream(doubles).sum() / doubles.length + 1) / 2).mapToDouble(d -> d).toArray());

                double[] rawLable = wav.get(i + 1);
                double[] lable = {(Arrays.stream(rawLable).sum() / rawLable.length + 1) / 2};
                BasicMLDataPair val = new BasicMLDataPair(dataSetEntry, new BasicMLData(lable));

                stmr.addData(i - histDepth, val);

                if (i % 5000 == 0)
                    System.out.println("Loaded " + i + ", input: " + val.getInput().size() + " label: " + val.getIdeal().size());
            }
            System.out.println("Done");
        }

        System.out.println("Cache size after streaming (" + cacheName + "): " + cache.size());
    }

    public static IgniteCache<Integer, MLDataPair> getOrCreate(Ignite ignite) {
        CacheConfiguration<Integer, MLDataPair> cfg = new CacheConfiguration<>();

        // Write to primary.
        cfg.setWriteSynchronizationMode(CacheWriteSynchronizationMode.PRIMARY_SYNC);

        // Atomic transactions only.
        cfg.setAtomicityMode(CacheAtomicityMode.ATOMIC);

        // No eviction.
        cfg.setEvictionPolicy(null);

        // No copying of values.
        cfg.setCopyOnRead(false);

        cfg.setOnheapCacheEnabled(true);

        // Cache is partitioned.
        cfg.setCacheMode(CacheMode.PARTITIONED);

        // Random cache name.
        cfg.setName(CACHE_NAME);

        return ignite.getOrCreateCache(cfg);
    }

}
