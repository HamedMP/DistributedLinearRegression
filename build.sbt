name := "lab1"

organization := "se.kth.spark"

version := "1.0"

scalaVersion := "2.11.1"

//resolvers += Resolver.mavenLocal
resolvers += "Kompics Snapshots" at "http://kompics.sics.se/maven/snapshotrepository/"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.1" // % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.1" // % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.1" // % "provided"
libraryDependencies += "org.log4s" %% "log4s" % "1.3.3" // % "provided"
libraryDependencies += "se.kth.spark" %% "lab1_lib" % "1.0-SNAPSHOT"

mainClass in assembly := Some("se.kth.spark.lab1.test.Main")

run in Compile := Defaults.runTask(fullClasspath in Compile, mainClass in(Compile, run), runner in(Compile, run)).evaluated

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)

libraryDependencies ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.2",

  // Native libraries are not included by default. add this if you want them (as of 0.7)
  // Native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.13.2",

  // The visualization library is distributed separately as well.
  // It depends on LGPL code
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)


resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"