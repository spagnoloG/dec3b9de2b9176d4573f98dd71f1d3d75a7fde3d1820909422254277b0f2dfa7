{
  description = "Opend3d developmen environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }@inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; };
        };

        fetchPypi = pkgs.python311Packages.fetchPypi;

        customPython = pkgs.python311.override {
          packageOverrides = self: super: {
            opencv4 = super.opencv4.override {
              enableGtk2 = true;
              gtk2 = pkgs.gtk2;
              enableFfmpeg = true;
            };
          };
        };

        open3d = pkgs.python311Packages.buildPythonPackage rec {
          pname = "open3d";
          version = "0.18.0";
          format = "wheel";

          src = pkgs.python311Packages.fetchPypi {
            pname = "open3d";
            version = "0.18.0";
            format = "wheel";
            sha256 = "sha256-jj0dGQCo9NlW9oGcJGx4CBclubCIj4VJ0qeknI2qEwM=";
            dist = "cp311";
            python = "cp311";
            abi = "cp311";
            platform = "manylinux_2_27_x86_64";
          };

          nativeBuildInputs = [ pkgs.autoPatchelfHook ];
          autoPatchelfIgnoreMissingDeps = [
            "libtorch_cuda_cpp.so"
            "libtorch_cuda_cu.so"
            "libtorch_cuda.so"
            "libc10_cuda.so"
          ];

          buildInputs = with pkgs; [
            cudaPackages.cudatoolkit
            stdenv.cc.cc.lib
            libusb.out
            libGL
            cudaPackages.cudatoolkit
            libtorch-bin
            libtensorflow
            expat
            xorg.libXfixes
            mesa
            xorg.libX11
            xorg.libXfixes
          ];

          propagatedBuildInputs = with pkgs.python311Packages; [
            nbformat
            numpy
            dash
            configargparse
            scikit-learn
            ipywidgets
            addict
            matplotlib
            pandas
            pyyaml
            tqdm
            pyquaternion
          ];

          postInstall = ''
            ln -s "${pkgs.llvm_10.lib}/lib/libLLVM-10.so" "$out/lib/libLLVM-10.so.1"

            rm $out/lib/python3.11/site-packages/open3d/libGL.so.1
            rm $out/lib/python3.11/site-packages/open3d/swrast_dri.so
            rm $out/lib/python3.11/site-packages/open3d/libgallium_dri.so
            rm $out/lib/python3.11/site-packages/open3d/kms_swrast_dri.so
            rm $out/lib/python3.11/site-packages/open3d/libEGL.so.1
          '';

        };

        pythonEnv = customPython.withPackages (ps:
          with ps; [
            matplotlib
            numpy
            scipy
            requests
            opencv4
            flake8
            black
            open3d
          ]);

      in {
        devShell = pkgs.mkShell {
          buildInputs =
            [ pythonEnv pkgs.python311Packages.pip pkgs.gtk2 pkgs.ffmpeg ];

          shellHook = ''
            echo "Welcome to the Python development environment."
          '';
        };
      });
}
