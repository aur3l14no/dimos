{
  description = "Voxel ray tracing native module for DimOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    dimos-rust = {
      url = "path:../../../../native/rust";
      flake = false;
    };
  };

  outputs =
    {
      crane,
      dimos-rust,
      flake-utils,
      nixpkgs,
      self,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        craneLib = crane.mkLib pkgs;
        lib = pkgs.lib;
        python = pkgs.python312;

        dimosRustRoot = toString dimos-rust;
        dimosRustSrc = lib.cleanSourceWith {
          src = dimos-rust;
          name = "dimos-rust-source";
          filter =
            path: type:
            let
              rel = lib.removePrefix "${dimosRustRoot}/" (toString path);
              inCrate = crate: rel == crate || lib.hasPrefix "${crate}/" rel;
            in
            rel == "Cargo.toml"
            || (
              (inCrate "dimos-module" || inCrate "dimos-module-macros")
              && craneLib.filterCargoSources path type
            );
        };
        crateSrc = pkgs.lib.fileset.toSource {
          root = ./.;
          fileset = craneLib.fileset.commonCargoSources ./.;
        };

        commonArgs = {
          pname = "voxel-ray-tracing";
          version = "0.1.0";
          src = crateSrc;

          cargoLock = ./Cargo.lock;
          cargoToml = ./Cargo.toml;
          RUSTFLAGS = pkgs.lib.optionalString pkgs.stdenv.isDarwin (
            "-C link-arg=-undefined -C link-arg=dynamic_lookup"
          );
          postPatch = ''
            substituteInPlace Cargo.toml \
              --replace-fail '../../../../native/rust/dimos-module' '${dimosRustSrc}/dimos-module'
          '';
          strictDeps = true;
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        package = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            doCheck = false;
            postInstall = ''
              sitePackages=$out/${python.sitePackages}
              mkdir -p "$sitePackages"
              cp target/release/libdimos_voxel_ray_tracing${pkgs.stdenv.hostPlatform.extensions.sharedLibrary} \
                "$sitePackages/dimos_voxel_ray_tracing.abi3.so"
              PYTHONPATH="$sitePackages" ${python.interpreter} -c \
                "from dimos_voxel_ray_tracing import VoxelRayMapper"
            '';
            meta.mainProgram = "voxel_ray_tracing";
          }
        );
        tests = craneLib.cargoTest (
          commonArgs
          // {
            inherit cargoArtifacts;
          }
        );
      in
      {
        packages.default = package;
        checks = {
          build = package;
          inherit tests;
        };
      }
    );
}
