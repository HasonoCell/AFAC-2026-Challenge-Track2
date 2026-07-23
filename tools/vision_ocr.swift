import CoreGraphics
import Foundation
import ImageIO
import Vision

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(
        Data("usage: vision_ocr.swift IMAGE\n".utf8)
    )
    exit(2)
}

let imageURL = URL(fileURLWithPath: CommandLine.arguments[1]) as CFURL
guard
    let source = CGImageSourceCreateWithURL(imageURL, nil),
    let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
else {
    FileHandle.standardError.write(Data("could not decode image\n".utf8))
    exit(1)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = ["zh-Hans", "en-US"]
request.usesLanguageCorrection = false

let handler = VNImageRequestHandler(cgImage: image, options: [:])
do {
    try handler.perform([request])
} catch {
    FileHandle.standardError.write(Data("Vision OCR failed: \(error)\n".utf8))
    exit(1)
}

let observations = (request.results ?? []).sorted { left, right in
    let verticalDelta = left.boundingBox.midY - right.boundingBox.midY
    if abs(verticalDelta) > 0.002 {
        return verticalDelta > 0
    }
    return left.boundingBox.minX < right.boundingBox.minX
}

for observation in observations {
    guard let candidate = observation.topCandidates(1).first else {
        continue
    }
    let box = observation.boundingBox
    let text = candidate.string
        .replacingOccurrences(of: "\t", with: " ")
        .replacingOccurrences(of: "\n", with: " ")
    print(
        String(
            format: "%.7f\t%.7f\t%.7f\t%.7f\t%.5f\t%@",
            box.minX,
            box.minY,
            box.width,
            box.height,
            candidate.confidence,
            text
        )
    )
}
