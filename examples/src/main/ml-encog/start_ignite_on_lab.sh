USER=user
TARGET_DIR=/home/user
BINARY_PATH=/home/user/IdeaProjects/gpr/target/bin/
BINARY_NAME=apache-ignite-fabric-2.3.0-SNAPSHOT-bin
CONFIG_NAME=example-ml-nn.xml
CONFIG_PATH=${BINARY_PATH}../../examples/config/${CONFIG_NAME}
JVM_OPTS="-Xmx5g"

hosts=( labx laby )

for host in "${hosts[@]}"
do
    # Stop ignite nodes. Very robust (stops all java, should be refactored)
    ssh ${host} 'kill -9 `pgrep -f java`'
    ssh ${host} "rm -rf ${TARGET_DIR}/${BINARY_NAME}"
    scp "${BINARY_PATH}/${BINARY_NAME}.zip" ${USER}@${host}:${TARGET_DIR}
    ssh ${host} unzip "${TARGET_DIR}/${BINARY_NAME}.zip"
    scp "${CONFIG_PATH}" ${USER}@${host}:${TARGET_DIR}
    ssh -n -f ${host} "cd ${TARGET_DIR}/${BINARY_NAME}/bin && ./ignite.sh ${TARGET_DIR}/${CONFIG_NAME} -J${JVM_OPTS} > ${TARGET_DIR}/node.log 2>${TARGET_DIR}/node.log &"
done
