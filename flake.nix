{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      supportedSystems = [
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (
        system:
        import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        }
      );
      pythonLibs = forAllSystems (
        system: with pkgs.${system}; [
          jupyter-all
          python312Packages.ipython
          python312Packages.numpy
          python312Packages.pandas
          python312Packages.matplotlib
          python312Packages.scikit-learn
          python312Packages.scipy
          python311Packages.torch
          # python311Packages.tensorflowWithCuda
        ]
      );
    in
    {
      devShells = forAllSystems (system: {
        default = pkgs.${system}.mkShell {
          packages =
            (with pkgs.${system}; [
              # Required for matplotlib to display with GTK4Cairo backend
              librsvg
              python312Packages.pygobject3
              python312Packages.pycairo
              gtk4
            ])
            ++ pythonLibs.${system};
          shellHook = ''
            export MPLBACKEND="GTK4Cairo"
          '';
        };
        jupyter = pkgs.${system}.mkShell {
          packages = pythonLibs.${system};
        };
        jupyterFHS =
          (pkgs.${system}.buildFHSUserEnv {
            name = "fhsJupyter";
            targetPkgs =
              pkgs:
              (with pkgs; [
                # cudatoolkit
                util-linux
                m4
                gperf
                unzip
                python313
                python313Packages.pip
                python313Packages.virtualenv
                python313Packages.numpy
		faiss
                python313Packages.faiss
                python313Packages.virtualenv
                python313Packages.uv
                python313Packages.ipython
                cmake
                ninja
                gcc
		git
		neovim
                pre-commit
                # linuxPackages.nvidia_x11
                # xorg.libXi
                # xorg.libXmu
                # freeglut
                # xorg.libXext
                # xorg.libX11
                # xorg.libXv
                # xorg.libXrandr
                zlib
                ncurses5
                stdenv.cc
                binutils
                # libGLU
                # libGL
                # cudaPackages.cudnn
              ]);
            runScript = "fish";
            profile = ''
              export EXTRA_CCFLAGS="-I/usr/include"
            '';
          }).env;
      });
      formatter = forAllSystems (system: pkgs.${system}.nixfmt-rfc-style);
    };
}
