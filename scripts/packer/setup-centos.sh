#!/bin/bash
# Setup script for CentOS GitHub Actions AMI
# Derived from:
# github.com/containers/ai-lab-recipes/blob/main/training/nvidia-bootc/Containerfile

set -euxo pipefail

DRIVER_VERSION="580.65.06"
# CUDA_VERSION is embedded in the driver "local repo" package

if [[ $(id -u) != "0" ]]; then
    echo "you must run this script as root."
    exit 1
fi

function configure_dnf {
  # Configure the DNF repos and options we need for CI.
  dnf -y install dnf-plugins-core
  dnf config-manager --save \
    --setopt=skip_missing_names_on_install=False \
    --setopt=install_weak_deps=False

  dnf -y install epel-release
  dnf -y install https://us.download.nvidia.com/tesla/$DRIVER_VERSION/nvidia-driver-local-repo-rhel9-$DRIVER_VERSION-1.0-1.x86_64.rpm
  # TODO: We might be able to use a nvidia.com yum repo instead of the local repo?
  # dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION_MAJOR}/${CUDA_REPO_ARCH}/cuda-rhel${OS_VERSION_MAJOR}.repo
}

function install_userland_packages {
  # CI tests in GH Actions will require these packages:
  dnf -y install nvtop podman skopeo git python3.12 python3.12-devel
}

function install_kernel_driver {
  # Install nvidia kernel driver.
  # DKMS will compile the nvidia.ko driver for all kernels for which we have installed a kernel-devel package.
  # By default, the "dnf module install" command will install the latest kernel-devel package that CentOS has published.
  dnf -y install "kernel-devel-$(uname -r)" gcc make dkms elfutils-libelf-devel  # also build for the currently-running kernel.
  # If we had configured a previous nvidia-driver version with DNF, reset it:
  dnf -y module reset nvidia-driver || true
  DRIVER_STREAM=$(echo $DRIVER_VERSION | cut -d. -f1)
  dnf -y module install nvidia-driver:${DRIVER_STREAM}-dkms # or use :latest-dkms after confirming available streams
}

function test_kernel_driver {
  # The nvidia driver DNF module (above) installs a dkms RPM.
  # That dkms RPM compiles and installs the nvidia.ko module.
  # List all the modules that dkms has compiled:
  dkms status || true
  # Load the module (ok if it’s already loaded or unavailable for this kernel):
  modprobe -q nvidia || true
  # If a GPU is present, verify userspace; otherwise, fail the job:
  nvidia-smi
}

function install_container_toolkit {
  # Install nvidia container toolkit.
  # When we pass GPU devices to a container (podman run --device nvidia.com/gpu=all), we use the nvidia CTK to do that.
  # See docs at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  curl -sSfL -o /etc/yum.repos.d/nvidia-container-toolkit.repo \
    https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
  dnf config-manager --enable nvidia-container-toolkit-experimental
  export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  dnf install -y \
      nvidia-container-toolkit-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1-${NVIDIA_CONTAINER_TOOLKIT_VERSION}
  # Verify it works:
  nvidia-ctk --version
  # When you boot a node, you must run:
  #   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  # This command scans your system for NVIDIA GPUs and creates a YAML file that lists the available devices.
}

configure_dnf
install_userland_packages
install_kernel_driver
test_kernel_driver
install_container_toolkit
