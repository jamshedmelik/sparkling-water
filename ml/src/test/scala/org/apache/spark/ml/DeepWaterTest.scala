/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.ml

import java.io.File

import hex.deepwater.{DeepWater, DeepWaterModel, DeepWaterParameters}
import org.apache.spark.SparkContext
import org.apache.spark.h2o.utils.SharedSparkTestContext
import org.junit.runner.RunWith
import org.junit.Assert
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner
import water.fvec.{Frame, NFSFileVec}
import water.parser.ParseDataset
import water.{DKV, Key}

@RunWith(classOf[JUnitRunner])
class DeepWaterTest extends FunSuite with SharedSparkTestContext {

  override def createSparkContext: SparkContext = new SparkContext("local[*]", "test-local", conf = defaultSparkConf)

  test("Prostate classification") {

    val p = new DeepWaterParameters
    p._backend = DeepWaterParameters.Backend.tensorflow

    val f = new File("/prostate.csv")
    val nfs: NFSFileVec = NFSFileVec.make(f)
    val tr: Frame = ParseDataset.parse(Key.make(), nfs._key)
    p._train = tr._key

    p._response_column = "CAPSULE"
    p._ignored_columns = Array[String]("ID")
    for (col <- Array[String]("RACE", "DPROS", "DCAPS", "CAPSULE", "GLEASON")) {
      val v = tr.remove(col)
      tr.add(col, v.toCategoricalVec)
      v.remove()
    }

    DKV.put(tr)
    p._seed = 1234
    p._epochs = 500
    val j = new DeepWater(p)

    val m: DeepWaterModel = j.trainModel.get

    println("DUPA")

    Assert.assertTrue(m._output._training_metrics.auc_obj._auc > 0.90)
  }

}
